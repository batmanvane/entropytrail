"""Temperature sweep comparison C1-C3 (frozen plan, METHODS.md; D-037), plus labelled descriptives.

    uv run --group analysis python scripts/compare_temperatures.py [--low qwen35-9b-bf16-mlx-t0p7-a31fc218=0.7]

Reads each condition's own results folder (T = 1.0: results/, other: results/<condition_id>/)
and writes results/temperature-sweep/summary.json. Holm with m = 3 per added temperature.
"""

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from entropytrail.analysis.sweep import REF, compare, stack
from entropytrail.storage import GRID_CONDITION, results_dir

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/Volumes/Data/entropytrail")
DESCRIBE = ["n_app_tokens", "fork_share", "late_js_surprise", "fork_surprise", "path_action_per_token",
            "functionality", "ui", "ux", "novelty_calibrated", "checklist_share", "renders", "ensemble_rarity"]


def load(cond: str) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    res = results_dir(DATA / cond, ROOT / "results")
    files = [res / f for f in ("outcomes.csv", "app-measures.csv", "ratings-claude.csv", "ensemble.csv")]
    out, meas, rat, ens = (pd.read_csv(f) for f in files)
    rat = rat[["prompt_id", "sample", "functionality", "ui", "ux", "novelty_calibrated", "checklist_passed",
               "checklist_total"]]
    apps = (out.merge(meas.drop(columns=["domain", "openness", "finish_reason", "html_extracted"]),
                      on=["prompt_id", "sample"], validate="1:1")
            .merge(rat, on=["prompt_id", "sample"], validate="1:1"))
    apps["checklist_share"] = apps["checklist_passed"] / apps["checklist_total"].replace(0, np.nan)
    apps["condition_id"] = cond
    ens["condition_id"] = cond
    return apps, ens, files


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if math.isnan(float(x)) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def describe(apps: pd.DataFrame) -> dict:
    g = apps.groupby("temperature")
    return {c: {str(t): {"mean": s.mean(), "median": s.median(), "n": int(s.notna().sum())}
                for t, s in g[c]} for c in DESCRIBE if c in apps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--low", action="append", default=None, help="condition_id=temperature (repeatable)")
    a = ap.parse_args()
    lows = dict(x.split("=") for x in (a.low or ["qwen35-9b-bf16-mlx-t0p7-a31fc218=0.7"]))
    ref_apps, ref_ens, inputs = load(GRID_CONDITION)
    summary = {"reference": {"condition_id": GRID_CONDITION, "temperature": REF}, "comparisons": {}}
    for cond, t in lows.items():
        t = float(t)
        apps, ens, files = load(cond)
        inputs += files
        A, E = stack({REF: ref_apps, t: apps}), stack({REF: ref_ens, t: ens})
        summary["comparisons"][str(t)] = {"condition_id": cond, **compare(A, E, t),
                                          "descriptive": {"label": "secondary, descriptive; not in the family",
                                                          **describe(A)}}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout)
    summary["provenance"] = {
        "git_commit": commit, "git_dirty": dirty, "decisions": ["D-030", "D-032", "D-034", "D-036", "D-037"],
        "inputs": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    out = ROOT / "results" / "temperature-sweep"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(_clean(summary), indent=2), encoding="utf-8")
    for t, c in summary["comparisons"].items():
        print(t, {k: (round(c[k]["estimate"], 4), round(c[k]["p_primary"], 6), c["holm"][k]["reject"])
                  for k in ("C1", "C2", "C3")})


if __name__ == "__main__":
    main()
