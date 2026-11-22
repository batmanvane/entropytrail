"""Collect fork importance per site into results/forks/importance-<plan>.csv (WP5).

    uv run python scripts/collect_forks.py --plan results/forks/plan-pilot.json [--root <condition dir>]

A site enters the table once at least one alternative and two same-token rollouts have a
complete eval/ folder (run `et eval --forks` first). Columns: see docs/DATA.md.
"""

import argparse
import csv
import json
from pathlib import Path

from entropytrail.evaluate import load_outcome
from entropytrail.forks import read_plan, rollout_dir
from entropytrail.importance import fork_importance

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["prompt_id", "sample", "domain", "openness", "pos", "kind", "H", "S", "section", "rel_pos",
          "n_alt", "n_same", "n_alt_renders", "n_same_renders", "alt_same", "noise_floor", "importance",
          "importance_incl_original"]


def _evaluated(d: Path) -> bool:
    return (d / "eval" / "eval.json").exists()


def collect(cond_dir: Path, plan: dict) -> list[dict]:
    rows = []
    for app in plan["apps"]:
        unit = Path(cond_dir) / app["prompt_id"] / f"{int(app['sample']):03d}"
        prompt = json.loads((unit / "meta.json").read_text()).get("prompt", {})
        original = load_outcome(unit) if _evaluated(unit) else None
        for s in app["sites"]:
            dirs = {v: rollout_dir(unit, s["pos"], v) for v in s["variants"]}
            alts = [load_outcome(d) for v, d in dirs.items() if v.startswith("alt") and _evaluated(d)]
            sames = [load_outcome(d) for v, d in dirs.items() if v.startswith("same") and _evaluated(d)]
            imp = fork_importance(alts, sames, original)
            if imp["importance"] is None:
                continue
            rows.append({
                "prompt_id": app["prompt_id"], "sample": int(app["sample"]),
                "domain": prompt.get("domain"), "openness": prompt.get("openness"),
                **{k: s[k] for k in ("pos", "kind", "H", "S", "section", "rel_pos")},
                "n_alt_renders": sum(o.renders for o in alts), "n_same_renders": sum(o.renders for o in sames),
                **imp,
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--root", default="/Volumes/Data/entropytrail/qwen35-9b-bf16-mlx-1195d88b", type=Path)
    a = ap.parse_args()
    rows = collect(a.root, read_plan(a.plan))
    out = ROOT / "results" / "forks" / f"importance-{a.plan.stem.removeprefix('plan-')}.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items() if k in FIELDS})
    print(f"{len(rows)} sites -> {out}")


if __name__ == "__main__":
    main()
