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


def render_app(html: str | None, out_dir: Path, *, browser, timeout_s: float = 15.0) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ms = int(timeout_s * 1000)
    rec = {
        "renders": False, "timeout": False,
        "n_page_errors": 0, "n_console_errors": 0, "page_errors": [], "console_errors": [],
        "n_page_errors_after_interaction": 0, "body_text_chars": 0, "blocked_requests": 0,
        "interaction_failed": [],
        "first_random": float("nan"), "viewport": list(VIEWPORT), "random_seed": RANDOM_SEED,
        "clock_start": CLOCK_START, "playwright": md.version("playwright"),
        "chromium": browser.version, "render_version": RENDER_VERSION,
    }
    shots = {"load": out_dir / "load.png", "interact": out_dir / "interact.png"}

    if html is not None:
        _run(html, rec, shots, browser, ms)
    for p in shots.values():
        if not p.exists():
            _blank(p)
    rec["page_errors"] = rec["page_errors"][:MAX_LISTED]
    rec["console_errors"] = rec["console_errors"][:MAX_LISTED]
    (out_dir / "render.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    return rec


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
            page.screenshot(path=str(shots["load"]), timeout=ms, animations="disabled")
            rec["renders"] = rec["n_page_errors"] == 0
            rec["body_text_chars"] = page.evaluate(
                "document.body ? document.body.innerText.length : 0")
            fr = page.evaluate("window.__et_first_random === undefined ? null : window.__et_first_random")
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
