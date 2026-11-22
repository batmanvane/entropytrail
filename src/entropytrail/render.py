"""Render one app in headless Chromium with a fixed, deterministic interaction (WP4, D-034).

The page runs with a seeded ``Math.random`` (mulberry32, seed 12345), Playwright's fake
clock (time only advances through ``clock.run_for``), blocking dialogs replaced by console
logging, and no network (every request other than the app's own file is aborted). Two
screenshots are taken: ``load.png`` after 1.5 s of page time and ``interact.png`` after a
fixed interaction. The same procedure is used for grid apps and fork rollouts.
"""
from __future__ import annotations

import importlib.metadata as md
import json
import os
import select
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

RENDER_VERSION = 1
VIEWPORT = (1024, 768)
RANDOM_SEED = 12345
CLOCK_INSTALL = "2026-01-01T11:59:59Z"
CLOCK_START = "2026-01-01T12:00:00Z"
MAX_LISTED = 20

INIT_SCRIPT = (
    "(()=>{"
    "window.alert=function(m){console.log('[alert]',m)};"
    "window.confirm=function(m){console.log('[confirm]',m);return true};"
    "window.prompt=function(m,d){console.log('[prompt]',m);return d===undefined?'':d};"
    f"let a={RANDOM_SEED}>>>0;"
    "const r=function(){a=(a+0x6D2B79F5)>>>0;let t=a;t=Math.imul(t^(t>>>15),t|1);"
    "t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296};"
    "let first=true;"
    "Math.random=function(){const v=r();if(first){first=false;window.__et_first_random=v}return v};"
    "})();"
)
INPUT_SELECTOR = "input[type=number]:visible, input[type=text]:visible, input:not([type]):visible"
KEYS = ("ArrowRight", "ArrowUp", "Space", "Enter")


def _blank(path: Path) -> None:
    from PIL import Image

    Image.new("RGB", VIEWPORT, "white").save(path)


def _base_record(chromium: str) -> dict:
    return {
        "renders": False, "timeout": False,
        "n_page_errors": 0, "n_console_errors": 0, "page_errors": [], "console_errors": [],
        "n_page_errors_after_interaction": 0, "body_text_chars": 0, "blocked_requests": 0,
        "interaction_failed": [], "hung_killed": False, "post_load_error": None,
        "first_random": float("nan"), "viewport": list(VIEWPORT), "random_seed": RANDOM_SEED,
        "clock_start": CLOCK_START, "playwright": md.version("playwright"),
        "chromium": chromium, "render_version": RENDER_VERSION,
    }


def _finish(rec: dict, out_dir: Path) -> dict:
    for name in ("load.png", "interact.png"):
        if not (out_dir / name).exists():
            _blank(out_dir / name)
    rec["page_errors"] = rec["page_errors"][:MAX_LISTED]
    rec["console_errors"] = rec["console_errors"][:MAX_LISTED]
    (out_dir / "render.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    return rec


def render_app(html: str | None, out_dir: Path, *, browser, timeout_s: float = 15.0) -> dict:
    """Render in-process. A page that busy-loops after load can block a Playwright call
    forever; use ``IsolatedRenderer`` for unattended batches."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ms = int(timeout_s * 1000)
    rec = _base_record(browser.version)
    shots = {"load": out_dir / "load.png", "interact": out_dir / "interact.png"}

    if html is not None:
        _run(html, rec, shots, browser, ms)
    return _finish(rec, out_dir)


def _run(html: str, rec: dict, shots: dict, browser, ms: int) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp) / "app.html"
        app.write_text(html, encoding="utf-8")
        url = app.as_uri()
        ctx = browser.new_context(viewport={"width": VIEWPORT[0], "height": VIEWPORT[1]},
                                  device_scale_factor=1)
        try:
            ctx.add_init_script(INIT_SCRIPT)

            def route(r):
                if r.request.url.startswith("file://"):
                    r.continue_()
                else:
                    rec["blocked_requests"] += 1
                    r.abort()

            ctx.route("**/*", route)
            page = ctx.new_page()
            page.set_default_timeout(ms)

            def on_error(e):
                rec["n_page_errors"] += 1
                rec["page_errors"].append(str(e)[:300])

            def on_console(m):
                if m.type == "error":
                    rec["n_console_errors"] += 1
                    rec["console_errors"].append(m.text[:300])

            page.on("pageerror", on_error)
            page.on("console", on_console)
            # install() alone lets fake time flow with real time; pause so page time only
            # advances through run_for (deterministic timers, rAF, Date, performance.now)
            page.clock.install(time=CLOCK_INSTALL)
            page.clock.pause_at(CLOCK_START)
            try:
                page.goto(url, wait_until="load", timeout=ms)
                page.clock.run_for(1500)
            except Exception:  # noqa: BLE001 - timeout or crash of the page
                rec["timeout"] = True
                return
            try:  # e.g. a page that navigates or reloads itself after load
                page.screenshot(path=str(shots["load"]), timeout=ms, animations="disabled")
                rec["body_text_chars"] = page.evaluate(
                    "document.body ? document.body.innerText.length : 0")
                fr = page.evaluate("window.__et_first_random === undefined ? null : window.__et_first_random")
            except Exception as e:  # noqa: BLE001
                rec["post_load_error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
                return
            rec["renders"] = rec["n_page_errors"] == 0
            if fr is not None:
                rec["first_random"] = fr
            errors_at_load = rec["n_page_errors"]
            rec["interaction_failed"] = _interact(page)
            try:
                page.screenshot(path=str(shots["interact"]), timeout=ms, animations="disabled")
            except Exception:  # noqa: BLE001 - page hung during the interaction
                rec["timeout"] = True
            rec["n_page_errors_after_interaction"] = rec["n_page_errors"] - errors_at_load
        finally:
            try:
                ctx.close()
            except Exception:  # noqa: BLE001
                pass


def _interact(page) -> list[str]:
    """Run the fixed interaction; each step is tried on its own. Returns failed step names."""
    short = 2000
    failed: list[str] = []

    def step(name, fn):
        try:
            fn()
        except Exception:  # noqa: BLE001 - e.g. element covered, page busy
            failed.append(name)

    def click():
        button = page.locator("button:visible").first
        canvas = page.locator("canvas:visible").first
        if button.count():
            button.click(timeout=short, no_wait_after=True)
        elif canvas.count():
            canvas.click(timeout=short, no_wait_after=True)

    def keys():
        for k in KEYS:
            page.keyboard.press(k)
            page.clock.run_for(150)

    def fill():
        field = page.locator(INPUT_SELECTOR).first
        if field.count():
            field.fill("42", timeout=short)

    step("click", click)
    step("keys", keys)
    step("fill", fill)
    step("wait", lambda: page.clock.run_for(1000))
    return failed


class IsolatedRenderer:
    """Renders each app in a persistent worker process with its own browser.

    If the worker does not answer within ``timeout_s + grace_s`` seconds (a page that
    busy-loops after load blocks Playwright calls that have no timeout, e.g. ``clock.run_for``),
    the worker's process group is killed, the unit gets a "hung" outcome (``timeout``
    true, ``renders`` false, ``hung_killed`` true, blank missing screenshots) and a new
    worker is started for the next unit. Use as a context manager; call like ``render_fn``.
    """

    def __init__(self, timeout_s: float = 15.0, grace_s: float = 20.0):
        self.timeout_s, self.grace_s = timeout_s, grace_s
        self.proc: subprocess.Popen | None = None
        self.chromium = ""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _start(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "entropytrail.render", str(self.timeout_s)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, start_new_session=True,
        )
        line = self._readline(60.0)
        if line is None:
            self._kill()
            raise RuntimeError("render worker did not start")
        self.chromium = json.loads(line)["chromium"]

    def _readline(self, timeout: float) -> str | None:
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if not ready:
            return None
        line = self.proc.stdout.readline()
        return line or None

    def _kill(self) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.proc.wait()
        self.proc = None

    def close(self) -> None:
        if self.proc is not None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                pass
            self._kill()

    def __call__(self, html: str | None, out_dir: Path) -> dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        with tempfile.TemporaryDirectory() as tmp:
            src = None
            if html is not None:
                src = Path(tmp) / "in.html"
                src.write_text(html, encoding="utf-8")
            self.proc.stdin.write(json.dumps({"html": str(src) if src else None, "out": str(out_dir)}) + "\n")
            self.proc.stdin.flush()
            line = self._readline(self.timeout_s + self.grace_s)
        if line is not None:
            rec = json.loads(line)
            if "error" in rec:
                raise RuntimeError(f"render worker error: {rec['error']}")
            return rec
        self._kill()
        rec = _base_record(self.chromium)
        rec.update(timeout=True, renders=False, hung_killed=True)
        return _finish(rec, out_dir)


def handle_request(req: dict, browser, timeout_s: float) -> dict:
    """One worker request; an exception becomes ``{"error": ...}`` so the caller can tell a
    crash (retried by the next ``et eval``) from a hang (recorded as hung)."""
    try:
        html = Path(req["html"]).read_text(encoding="utf-8") if req["html"] else None
        return render_app(html, Path(req["out"]), browser=browser, timeout_s=timeout_s)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"}


def _worker(timeout_s: float) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        print(json.dumps({"chromium": browser.version}), flush=True)
        for line in sys.stdin:
            if not browser.is_connected():
                browser = p.chromium.launch()
            print(json.dumps(handle_request(json.loads(line), browser, timeout_s)), flush=True)
        browser.close()


if __name__ == "__main__":
    _worker(float(sys.argv[1]) if len(sys.argv) > 1 else 15.0)
