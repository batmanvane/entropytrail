"""Automatic pre-check before the Chrome ratings (protocol v1.2, D-031).

    uv run python scripts/precheck_apps.py [--root /Volumes/Data/entropytrail/<condition_id>] [--prompt ID]

For every unit with an app.html and no rating-claude.json yet, all inline <script> blocks
are concatenated and parsed with `node --check`. A unit whose JavaScript does not parse can
never run, so it gets an automatic rating (functionality 1, ux 1, ui and novelty null,
rater "auto-precheck") and is skipped by the Chrome raters. Units that parse are left for
the Chrome test. A unit without app.html (no HTML extracted) is rated the same way.
Prints one line per prompt: prompt, parsed, syntax errors, already rated.
"""

import argparse
import json
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

NODE = "/opt/homebrew/bin/node"
SCRIPT = re.compile(r"<script\b[^>]*>(.*?)</script>", re.S | re.I)


def extract_js(html: str) -> str:
    return "\n;\n".join(SCRIPT.findall(html))


def syntax_error(js: str, node: str = NODE) -> str | None:
    """Return node's first error line if `js` does not parse, else None."""
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(js)
        path = f.name
    try:
        r = subprocess.run([node, "--check", path], capture_output=True, text=True)
    finally:
        Path(path).unlink(missing_ok=True)
    if r.returncode == 0:
        return None
    lines = [ln for ln in r.stderr.splitlines() if "Error" in ln]
    return lines[0].strip() if lines else "syntax error"


def auto_rating(reason: str) -> dict:
    return {
        "rater": "auto-precheck",
        "rater_model": None,
        "protocol": "v1.2",
        "rated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "functionality": 1,
        "ui": None,
        "ux": 1,
        "novelty": None,
        "checklist": {},
        "console_errors": 1,
        "what_it_is": None,
        "note": f"Automatic pre-check: {reason}. The app cannot run; not tested in Chrome.",
    }


def precheck_unit(unit: Path) -> str:
    """Return 'rated' (already), 'parsed', or 'syntax' (auto-rated now)."""
    if (unit / "rating-claude.json").exists():
        return "rated"
    app = unit / "app.html"
    if not app.exists():
        if not (unit / "meta.json").exists():
            return "pending"
        reason = "no HTML could be extracted from the model output"
    else:
        err = syntax_error(extract_js(app.read_text(encoding="utf-8")))
        if err is None:
            return "parsed"
        reason = f"JavaScript does not parse ({err})"
    (unit / "rating-claude.json").write_text(json.dumps(auto_rating(reason), indent=2, ensure_ascii=False))
    return "syntax"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Volumes/Data/entropytrail/qwen35-9b-bf16-mlx-1195d88b")
    ap.add_argument("--prompt")
    a = ap.parse_args()
    root = Path(a.root)
    prompts = [a.prompt] if a.prompt else sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("smoke"))
    for p in prompts:
        counts = {"parsed": 0, "syntax": 0, "rated": 0, "pending": 0}
        for unit in sorted(u for u in (root / p).iterdir() if re.fullmatch(r"\d{3}", u.name)):
            counts[precheck_unit(unit)] += 1
        print(f"{p}  parsed {counts['parsed']}  syntax-error {counts['syntax']}  already-rated {counts['rated']}")


if __name__ == "__main__":
    main()
