"""Download the human ratings (D-043) from the ratings Worker into results/ratings-human.csv.

    uv run python scripts/export_ratings.py

The export key is read from ~/.config/entropytrail/export_key (never committed). Raters are
anonymous random ids; the file joins to apps via results/rating-sample-ids.csv.
"""
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://entropytrail-ratings.entropytrail.workers.dev/export"

key = (Path.home() / ".config" / "entropytrail" / "export_key").read_text().strip()
req = urllib.request.Request(URL, headers={"Authorization": f"Bearer {key}", "User-Agent": "entropytrail-export"})
data = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
out = ROOT / "results" / "ratings-human.csv"
out.write_text(data, encoding="utf-8")
print(f"{out}: {len(data.splitlines()) - 1} ratings")
