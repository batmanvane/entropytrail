"""Per-app landscape measures of the grid into results/app-measures.csv (WP6, C1).

    uv run --group analysis python scripts/collect_measures.py [--root <condition dir>]

Apps without extracted HTML get a row with empty measures (frozen plan: kept as "no HTML",
excluded from token-level measures). Definitions: docs/METHODS.md.
"""

import argparse
import csv
import json
import math
from pathlib import Path

from entropytrail.analysis.measures import app_measures, app_tokens
from entropytrail.evaluate import find_units
from entropytrail.storage import results_dir

ROOT = Path(__file__).resolve().parents[1]
ID_FIELDS = ["prompt_id", "sample", "domain", "openness", "finish_reason", "html_extracted"]
MEASURES = ["n_app_tokens", "mean_H", "mean_S", "share_H_lt_0_1", "n_forks", "fork_share", "comment_fork_share",
            "fork_surprise", "late_js_surprise", "path_action", "path_action_per_token", "H_per_char",
            "H_per_line", "median_relpos_top10_forks"]


def collect(cond_dir: Path) -> list[dict]:
    rows = []
    for u in find_units(cond_dir, prompt_ids=None, forks=False):
        meta = json.loads((u / "meta.json").read_text())
        p = meta.get("prompt", {})
        row = {"prompt_id": u.parent.name, "sample": u.name, "domain": p.get("domain"),
               "openness": p.get("openness"), "finish_reason": meta.get("finish_reason"),
               "html_extracted": meta.get("html_extracted")}
        app = u / "app.html"
        if meta.get("html_extracted") and app.exists():
            html = app.read_text(encoding="utf-8")
            m = app_measures(app_tokens(u), len(html), html.count("\n") + 1)
            row.update({k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in m.items()})
        else:
            row.update({k: None for k in MEASURES})
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Volumes/Data/entropytrail/qwen35-9b-bf16-mlx-1195d88b", type=Path)
    ap.add_argument("--results", default=ROOT / "results", type=Path)
    a = ap.parse_args()
    rows = collect(a.root)
    out = results_dir(a.root, a.results) / "app-measures.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ID_FIELDS + MEASURES)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"{len(rows)} apps -> {out}")


if __name__ == "__main__":
    main()
