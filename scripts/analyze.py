"""Run the primary tests of H1-H5 on what exists and write results/analysis/summary.json (WP6).

    uv run --group analysis python scripts/analyze.py [--root <condition dir>]

Refuses to run until docs/DECISIONS.md has D-036 (Robert's answers to the open readings
R1-R3 of the frozen plan, plan 2 Part C): the confirmatory tests must not be computed
before their reading is fixed. H3 runs when results/forks/importance-*.csv exist; H4/H5
primary only when results/ratings-human.csv exists (WP7). Holm uses m = 5 throughout and
the summary is flagged interim until all five p-values exist. Claude ratings (D-030) go
into a separate exploratory file and never into Holm.
"""

import argparse
import hashlib
import importlib.metadata as md
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from entropytrail.analysis.measures import app_tokens
from entropytrail.analysis.tests_h import h1, h2, h3, holm, rating_test, renders_test

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis"


def require_readings(decisions: Path) -> None:
    if "## D-036" not in Path(decisions).read_text(encoding="utf-8"):
        raise SystemExit("refusing to run: D-036 (answers to R1-R3 of plan 2) is not in DECISIONS.md yet")


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items() if not isinstance(v, pd.DataFrame)}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(float(x)) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:  # noqa: BLE001 - e.g. too few sites for a sensitivity fit
        return {"status": f"failed: {type(e).__name__}: {e}"}


def build_summary(meas: pd.DataFrame, ens: pd.DataFrame, frames: dict, importance: pd.DataFrame | None,
                  human: pd.DataFrame | None, ens_sensitivity: pd.DataFrame | None = None) -> dict:
    s: dict = {}
    trunc = set(zip(meas.loc[meas["finish_reason"] == "max_tokens", "prompt_id"],
                    meas.loc[meas["finish_reason"] == "max_tokens", "sample"]))

    r1 = h1({f"{p}/{q}": f for (p, q), f in frames.items()})
    r1["sensitivity"] = _clean(h1({f"{p}/{q}": f for (p, q), f in frames.items() if (p, q) not in trunc}))
    s["H1"] = {"status": "done", **r1}

    r2 = h2(ens, meas)
    r2["sensitivity"] = (_safe(h2, ens_sensitivity, meas[meas["finish_reason"] != "max_tokens"])
                         if ens_sensitivity is not None else {"status": "not provided"})
    s["H2"] = {"status": "done", **r2}

    if importance is not None and len(importance):
        r3 = h3(importance)
        parent_ok = [(p, q) not in trunc for p, q in zip(importance["prompt_id"], importance["sample"])]
        r3["sensitivity"] = {"H_ge_2": _safe(h3, importance[importance["H"] >= 2]),
                             "no_truncated_parent": _safe(h3, importance[parent_ok])}
        s["H3"] = {"status": "done", **r3}
    else:
        s["H3"] = {"status": "waiting for fork experiments (WP5)"}

    if human is not None and len(human):
        d = meas.merge(human, on=["prompt_id", "sample"])
        nt = d[d["finish_reason"] != "max_tokens"]
        s["H4"] = {"status": "done", **rating_test(d, "usefulness_mean", "late_js_surprise", "-"),
                   "sensitivity": _safe(rating_test, nt, "usefulness_mean", "late_js_surprise", "-")}
        s["H5"] = {"status": "done", **rating_test(d, "novelty_mean", "fork_surprise", "+"),
                   "sensitivity": _safe(rating_test, nt, "novelty_mean", "fork_surprise", "+")}
    else:
        s["H4"] = {"status": "waiting for human ratings (WP7)"}
        s["H5"] = {"status": "waiting for human ratings (WP7)"}

    pv = {h: s[h]["p_primary"] for h in ("H1", "H2", "H3", "H4", "H5") if s[h]["status"] == "done"}
    s["holm"] = holm(pv)
    s["interim"] = len(pv) < 5
    return _clean(s)


def secondary(meas: pd.DataFrame, outcomes: pd.DataFrame) -> dict:
    """Secondary analyses that need no human ratings (fixed in plan 2, C4)."""
    d = meas.merge(outcomes[["prompt_id", "sample", "renders", "ensemble_rarity"]], on=["prompt_id", "sample"])
    return _clean({
        "H4_renders": _safe(renders_test, d, "late_js_surprise"),
        "H5_rarity_outcome": _safe(rating_test, d, "ensemble_rarity", "fork_surprise", "+", None),
    })


def exploratory_claude(meas: pd.DataFrame, claude: pd.DataFrame) -> dict:
    """Exploratory, labelled: the H4/H5 models with Claude ratings (D-030). Never in Holm."""
    c = claude[claude["rater"] != "auto-precheck"][["prompt_id", "sample", "functionality", "novelty_calibrated"]]
    d = meas.merge(c, on=["prompt_id", "sample"])
    return _clean({
        "label": "exploratory - Claude ratings, not human ratings; not part of the Holm family",
        "H4_functionality": _safe(rating_test, d, "functionality", "late_js_surprise", "-", None),
        "H5_novelty_calibrated": _safe(rating_test, d, "novelty_calibrated", "fork_surprise", "+", None),
    })


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _ens_without_truncated(root: Path, meas: pd.DataFrame) -> pd.DataFrame:
    from entropytrail.evaluate import load_outcome
    from entropytrail.outcome import ensemble_spread

    rows = []
    for (pid, dom, op), g in meas.groupby(["prompt_id", "domain", "openness"]):
        keep = g[g["finish_reason"] != "max_tokens"]
        outs = [load_outcome(root / pid / f"{int(q):03d}") for q in keep["sample"]]
        rows.append({"prompt_id": pid, "domain": dom, "openness": op,
                     "ensemble_spread": ensemble_spread(outs) if len(outs) > 1 else np.nan})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Volumes/Data/entropytrail/qwen35-9b-bf16-mlx-1195d88b", type=Path)
    a = ap.parse_args()
    require_readings(ROOT / "docs" / "DECISIONS.md")
    inputs = [ROOT / "results" / "app-measures.csv", ROOT / "results" / "ensemble.csv",
              ROOT / "results" / "outcomes.csv"]
    meas = pd.read_csv(inputs[0], dtype={"sample": int})
    ens = pd.read_csv(inputs[1])
    outcomes = pd.read_csv(inputs[2], dtype={"sample": int})
    frames = {(p, q): app_tokens(a.root / p / f"{q:03d}") for p, q in zip(meas["prompt_id"], meas["sample"])}
    imp_files = sorted((ROOT / "results" / "forks").glob("importance-*.csv"))
    importance = pd.concat([pd.read_csv(f) for f in imp_files]) if imp_files else None
    human_f = ROOT / "results" / "ratings-human.csv"
    human = pd.read_csv(human_f, dtype={"sample": int}) if human_f.exists() else None
    inputs += imp_files + ([human_f] if human is not None else [])

    summary = build_summary(meas, ens, frames, importance, human, _ens_without_truncated(a.root, meas))
    summary["secondary"] = secondary(meas, outcomes)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout)
    summary["provenance"] = {
        "git_commit": commit, "git_dirty": dirty,
        "inputs": {str(p.relative_to(ROOT)): _sha(p) for p in inputs},
        "packages": {k: md.version(k) for k in ("numpy", "pandas", "scipy", "statsmodels", "diptest")},
        "decisions": ["D-028", "D-029", "D-034", "D-035", "D-036"],
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    named = {f"{p}/{q:03d}": f for (p, q), f in frames.items()}
    h1(named)["per_app"].to_csv(OUT / "h1-per-app.csv", index=False)
    claude_f = ROOT / "results" / "ratings-claude.csv"
    if claude_f.exists():
        claude = pd.read_csv(claude_f, dtype={"sample": int})
        (OUT / "exploratory-claude-ratings.json").write_text(
            json.dumps(exploratory_claude(meas, claude), indent=2), encoding="utf-8")
    print(json.dumps({h: summary[h].get("status") for h in ("H1", "H2", "H3", "H4", "H5")}))


if __name__ == "__main__":
    main()
