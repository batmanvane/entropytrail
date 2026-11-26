"""Serve generated apps for rating, with a dialog shim injected.

    uv run python scripts/serve_apps.py [--port 8770] [--root /Volumes/Data/entropytrail/<condition_id>] [--blind-key key.json]

GET /<prompt_id>/<sample>/ returns that unit's app.html with a small script inserted at the
very start, which replaces alert/confirm/prompt by console logging (browser dialogs would
block the Claude in Chrome extension). Only app.html is served; tokens, meta and ratings are
not reachable through the server (rater blindness, docs/RATING-CLAUDE.md). Binds to 127.0.0.1.
"""

import argparse
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SHIM = (
    "<script>/* rating shim: no blocking dialogs */"
    "window.alert=function(m){console.log('[alert]',m)};"
    "window.confirm=function(m){console.log('[confirm]',m);return true};"
    "window.prompt=function(m,d){console.log('[prompt]',m);return d===undefined?'':d};"
    "</script>"
)
UNIT = re.compile(r"^/([a-z0-9-]+)/(\d{3})/?(?:app\.html)?$")


def storage_guard(unit: str) -> str:
    """All apps share one origin; clear web storage when a different unit is loaded,
    keep it on a reload of the same unit (so persistence can still be tested)."""
    return (
        "<script>/* storage guard */try{if(localStorage.getItem('__et_unit')!=='" + unit + "')"
        "{localStorage.clear();sessionStorage.clear();localStorage.setItem('__et_unit','" + unit + "')}}catch(e){}</script>"
    )


def inject(html: str, unit: str = "") -> str:
    """Insert SHIM (and the storage guard) before any other script: after <head> if present."""
    pre = SHIM + (storage_guard(unit) if unit else "")
    m = re.search(r"<head[^>]*>", html, re.I)
    if m:
        return html[: m.end()] + pre + html[m.end():]
    return pre + html


def make_handler(root: Path, blind: dict[str, dict[str, str]] | None = None):
    """Serve <root>/<prompt>/<sample>/app.html, or with ``blind`` ({prompt: {blind id: unit dir}})
    the unit behind a blind id (cross-temperature novelty pass, D-041)."""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            m = UNIT.match(self.path.split("?")[0])
            if blind is not None:
                unit = blind.get(m.group(1), {}).get(m.group(2)) if m else None
                app = Path(unit) / "app.html" if unit else None
            else:
                app = root / m.group(1) / m.group(2) / "app.html" if m else None
            if not app or not app.is_file():
                self.send_error(404)
                return
            body = inject(app.read_text(encoding="utf-8"), f"{m.group(1)}/{m.group(2)}").encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--root", default="/Volumes/Data/entropytrail/qwen35-9b-bf16-mlx-1195d88b")
    ap.add_argument("--blind-key", help="key.json of the cross-temperature novelty pass (D-041); serves blind ids")
    a = ap.parse_args()
    blind = None
    if a.blind_key:
        import json

        from entropytrail.novelty_cross import server_map

        blind = server_map(json.loads(Path(a.blind_key).read_text()))
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(Path(a.root), blind)).serve_forever()


if __name__ == "__main__":
    main()
