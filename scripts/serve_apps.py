"""Serve generated apps for rating, with a dialog shim injected.

    uv run python scripts/serve_apps.py [--port 8770] [--root /Volumes/Data/entropytrail/<condition_id>]

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


def inject(html: str) -> str:
    """Insert SHIM before any other script: after <head> if present, else at the start."""
    m = re.search(r"<head[^>]*>", html, re.I)
    if m:
        return html[: m.end()] + SHIM + html[m.end():]
    return SHIM + html


def make_handler(root: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            m = UNIT.match(self.path.split("?")[0])
            app = root / m.group(1) / m.group(2) / "app.html" if m else None
            if not app or not app.is_file():
                self.send_error(404)
                return
            body = inject(app.read_text(encoding="utf-8")).encode("utf-8")
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
    a = ap.parse_args()
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(Path(a.root))).serve_forever()


if __name__ == "__main__":
    main()
