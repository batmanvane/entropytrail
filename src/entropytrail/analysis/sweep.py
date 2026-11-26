"""Temperature sweep comparison C1-C3 (frozen plan, METHODS.md "Temperature sweep comparison", D-037).

One family per added temperature: the reference T = 1.0 against one lower temperature ``low``.
``lowT`` = 1 for apps of the lower temperature, so its coefficient is the effect of lowering T.
Design coding as D-036 (``loose`` + ``C(domain)``). Holm with m = 3 across C1-C3.
"""
from __future__ import annotations

import warnings

import pandas as pd

from entropytrail.analysis.tests_h import design, holm, one_sided

REF = 1.0


def stack(tables: dict[float, pd.DataFrame]) -> pd.DataFrame:
    """Stack per-temperature tables, refusing a table that mixes conditions or repeats a unit."""
    parts = []
    for temp, df in tables.items():
        if "condition_id" in df and df["condition_id"].nunique() > 1:
            raise ValueError(f"table for T = {temp} mixes conditions: {sorted(df['condition_id'].unique())}")
        key = ["prompt_id", "sample"] if "sample" in df else ["prompt_id"]
        if df.duplicated(key).any():
            raise ValueError(f"table for T = {temp} has repeated units ({key})")
        parts.append(df.assign(temperature=float(temp)))
    return pd.concat(parts, ignore_index=True)


def _low(d: pd.DataFrame, low: float) -> pd.DataFrame:
    d = d[d["temperature"].isin([REF, low])].copy()
    if set(d["temperature"]) != {REF, low}:
        raise ValueError(f"need apps of T = {REF} and T = {low}")
    d["lowT"] = (d["temperature"] == low).astype(int)
    return design(d)


def _result(fit, d, sign, model, **extra) -> dict:
    coef, p_two = float(fit.params["lowT"]), float(fit.pvalues["lowT"])
    lo, hi = (float(x) for x in fit.conf_int().loc["lowT"])
    return {"n": len(d), "n_prompts": int(d["prompt_id"].nunique()), "estimate": coef, "ci": [lo, hi],
            "p_two_sided": p_two, "p_primary": one_sided(coef, p_two, sign), "sign": sign, "model": model, **extra}


def c1_renders(apps: pd.DataFrame, low: float) -> dict:
    """C1: logit renders ~ lowT + loose + C(domain), SE cluster-robust by prompt; expected > 0."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    d = _low(apps, low)
    d["renders"] = d["renders"].astype(int)
    fit = smf.glm("renders ~ lowT + loose + C(domain)", d, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["prompt_id"])[0]})
    share = d.groupby("temperature")["renders"].mean()
    return _result(fit, d, "+", "Logit renders ~ lowT + loose + C(domain), cluster-robust SE by prompt",
                   share={str(k): float(v) for k, v in share.items()})


def c2_spread(ens: pd.DataFrame, low: float) -> dict:
    """C2: per-prompt screen spread ~ lowT + C(prompt) (paired by prompt); expected < 0."""
    import statsmodels.formula.api as smf

    d = _low(ens, low).dropna(subset=["ensemble_spread"])
    fit = smf.ols("ensemble_spread ~ lowT + C(prompt_id)", d).fit()
    mean = d.groupby("temperature")["ensemble_spread"].mean()
    return _result(fit, d, "-", "OLS ensemble_spread ~ lowT + C(prompt), paired by prompt",
                   mean={str(k): float(v) for k, v in mean.items()})


def c3_novelty(apps: pd.DataFrame, low: float) -> dict:
    """C3: MixedLM novelty_calibrated ~ lowT + loose + C(domain) + (1|prompt); expected < 0."""
    import statsmodels.formula.api as smf

    d = _low(apps, low).dropna(subset=["novelty_calibrated"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.mixedlm("novelty_calibrated ~ lowT + loose + C(domain)", d, groups=d["prompt_id"]).fit()
    mean = d.groupby("temperature")["novelty_calibrated"].mean()
    return _result(fit, d, "-", "MixedLM novelty_calibrated ~ lowT + loose + C(domain) + (1|prompt)",
                   mean={str(k): float(v) for k, v in mean.items()})


def compare(apps: pd.DataFrame, ens: pd.DataFrame, low: float) -> dict:
    res = {"C1": c1_renders(apps, low), "C2": c2_spread(ens, low), "C3": c3_novelty(apps, low)}
    res["holm"] = holm({k: res[k]["p_primary"] for k in ("C1", "C2", "C3")}, m=3)
    return res
