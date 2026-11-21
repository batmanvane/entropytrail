"""Collect all rating-claude.json files of a condition into results/ratings-claude.csv.

    uv run python scripts/collect_ratings.py [--root /Volumes/Data/entropytrail/<condition_id>]
"""

import argparse
import csv
import json
import re
from pathlib import Path

FIELDS = ["prompt_id", "sample", "rater", "rater_model", "protocol", "rated_at", "functionality", "ui", "ux",
          "novelty", "novelty_calibrated", "console_errors", "checklist_passed", "checklist_total", "what_it_is", "note"]


def rows(root: Path):
    for f in sorted(root.glob("*/[0-9][0-9][0-9]/rating-claude.json")):
        prompt, sample = f.parts[-3], f.parts[-2]
        if prompt.startswith("smoke"):
            continue
        r = json.loads(f.read_text())
        cl = r.get("checklist") or {}
        yield {**{k: r.get(k) for k in FIELDS if k in r}, "prompt_id": prompt, "sample": sample,
               "checklist_passed": sum(bool(v) for v in cl.values()), "checklist_total": len(cl)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Volumes/Data/entropytrail/qwen35-9b-bf16-mlx-1195d88b")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "ratings-claude.csv"))
    a = ap.parse_args()
    data = list(rows(Path(a.root)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(data)
    print(f"{len(data)} ratings -> {a.out}")


if __name__ == "__main__":
    main()
