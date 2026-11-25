"""Blind novelty calibration across temperatures (D-041, exploratory).

All apps of one prompt from every temperature are pooled, shuffled (seeded per prompt) and
served under blind ids 000, 001, ... by ``scripts/serve_apps.py --blind-map``; one agent per
prompt scores them on one 1-5 scale. The key (blind id -> condition, sample) stays outside
the agent's view; ``collect`` maps the scores back and ``analyse`` fits
``novelty_cross ~ C(temperature, ref 1.0) + loose + C(domain) + (1|prompt)``.
"""
from __future__ import annotations

import json
import random
import warnings
from pathlib import Path

import pandas as pd

from entropytrail.analysis.tests_h import design, one_sided

REF = 1.0
LABEL = ("exploratory (D-041) - cross-temperature blind novelty by Claude; defined after the frozen C3 "
         "result was seen; not part of any Holm family")


def build_key(conditions: dict[float, Path], prompt_ids: list[str], seed: int = 41) -> dict:
    prompts = {}
    for pid in prompt_ids:
        items = []
        for t, cdir in sorted(conditions.items(), reverse=True):
            for u in sorted((Path(cdir) / pid).glob("[0-9][0-9][0-9]")):
                if (u / "app.html").is_file():
                    items.append({"temperature": float(t), "condition_id": Path(cdir).name, "sample": int(u.name),
                                  "path": str(u)})
        random.Random(f"{seed}:{pid}").shuffle(items)
        prompts[pid] = [{"blind": f"{i:03d}", **it} for i, it in enumerate(items)]
    return {"version": 1, "seed": seed, "temperatures": sorted(float(t) for t in conditions), "prompts": prompts}


def server_map(key: dict) -> dict[str, dict[str, str]]:
    """{prompt: {blind id: unit dir}} for the app server."""
    return {pid: {it["blind"]: it["path"] for it in items} for pid, items in key["prompts"].items()}


def collect(key: dict, scores_dir: Path) -> pd.DataFrame:
    rows = []
    for pid, items in key["prompts"].items():
        f = Path(scores_dir) / f"{pid}.json"
        if not f.exists():
            raise ValueError(f"no scores for {pid}")
        sc = json.loads(f.read_text())["scores"]
        missing = sorted({it["blind"] for it in items} - set(sc))
        if missing:
            raise ValueError(f"{pid}: no score for blind ids {missing}")
        for it in items:
            s = sc[it["blind"]]
            rows.append({"prompt_id": pid, "blind": it["blind"], "temperature": it["temperature"],
                         "condition_id": it["condition_id"], "sample": it["sample"],
                         "novelty_cross": s.get("novelty"), "idea": s.get("idea", "")})
    return pd.DataFrame(rows)


def analyse(df: pd.DataFrame) -> dict:
    import statsmodels.formula.api as smf

    d = design(df.dropna(subset=["novelty_cross"]))
    lows = sorted((t for t in d["temperature"].unique() if t != REF), reverse=True)
    for t in lows:
        d[f"T{t}"] = (d["temperature"] == t).astype(int)
    names = {t: f"T{t}" for t in lows}
    terms = " + ".join(f"Q('{n}')" for n in names.values())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.mixedlm(f"novelty_cross ~ {terms} + loose + C(domain)", d, groups=d["prompt_id"]).fit()
    effects = {}
    for t, n in names.items():
        k = f"Q('{n}')"
        coef, p = float(fit.params[k]), float(fit.pvalues[k])
        lo, hi = (float(x) for x in fit.conf_int().loc[k])
        effects[str(t)] = {"estimate": coef, "ci": [lo, hi], "p_two_sided": p, "p_one_sided": one_sided(coef, p, "-")}
    means = d.groupby("temperature")["novelty_cross"].agg(["mean", "count"])
    return {"label": LABEL, "n": len(d), "n_prompts": int(d["prompt_id"].nunique()), "effects": effects,
            "means": {str(t): {"mean": float(r["mean"]), "n": int(r["count"])} for t, r in means.iterrows()},
            "model": "MixedLM novelty_cross ~ temperature (ref 1.0) + loose + C(domain) + (1|prompt)"}
