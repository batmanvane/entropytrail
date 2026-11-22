"""Primary tests of H1-H5 (frozen analysis plan, METHODS.md; plan 2 Part C; D-036).

Design coding for every model: ``loose`` (1 for loose prompts, 0 for tight and free) plus
``C(domain)`` with levels education/game/tool/free. Free prompts are the only "open" ones,
so openness and domain cannot both enter as factors; with this coding the loose-vs-tight
contrast is estimated within the three domains and all 21 prompts stay in every model.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05


def design(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["loose"] = (out["openness"] == "loose").astype(int)
    return out


def one_sided(coef: float, p_two: float, sign: str) -> float:
    """One-sided p in direction ``sign`` ('+' or '-') from a two-sided Wald p-value."""
    right = coef > 0 if sign == "+" else coef < 0
    return p_two / 2 if right else 1 - p_two / 2


# --- H1 ---------------------------------------------------------------------------------

def _dip_p(H: np.ndarray) -> float:
    import diptest

    x = np.log10(np.asarray(H, np.float64) + 1e-3)
    return float(diptest.diptest(x)[1])  # p-value by interpolation of the package's table


def h1(token_frames: dict[str, pd.DataFrame]) -> dict:
    rows = []
    for app, df in token_frames.items():
        code = df[df["is_code"]]
        rows.append({"app": app, "n_tokens": len(df), "dip_p": _dip_p(df["H"]),
                     "dip_p_code_only": _dip_p(code["H"]) if len(code) >= 4 else np.nan})
    per = pd.DataFrame(rows)
    rej = per["dip_p"] < ALPHA
    k, n = int(rej.sum()), len(per)
    return {
        "n": n, "share_rejecting": k / n if n else np.nan,
        "support_frozen": bool(n and k / n >= 0.80),
        # R1: one-sided binomial p (H0: true share <= 0.80); used only for Holm
        "p_primary": float(stats.binomtest(k, n, 0.80, alternative="greater").pvalue) if n else np.nan,
        "share_rejecting_code_only": float((per["dip_p_code_only"] < ALPHA).mean()) if n else np.nan,
        "dip_p_method": "diptest.diptest table interpolation",
        "per_app": per,
    }


# --- H2 ---------------------------------------------------------------------------------

def h2(ensemble: pd.DataFrame, measures: pd.DataFrame) -> dict:
    import statsmodels.formula.api as smf

    d = design(ensemble)
    fit = smf.ols("ensemble_spread ~ loose + C(domain)", data=d).fit()
    coef, p_two = float(fit.params["loose"]), float(fit.pvalues["loose"])
    lo, hi = (float(x) for x in fit.conf_int().loc["loose"])
    out = {"n": len(d), "estimate": coef, "ci": [lo, hi], "p_two_sided": p_two,
           "p_primary": one_sided(coef, p_two, "+"), "model": "OLS ensemble_spread ~ loose + C(domain)"}
    m = design(measures).dropna(subset=["median_relpos_top10_forks"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mf = smf.mixedlm("median_relpos_top10_forks ~ loose + C(domain)", m, groups=m["prompt_id"]).fit()
    out["secondary_relpos"] = {
        "n": len(m), "estimate": float(mf.params["loose"]), "p_two_sided": float(mf.pvalues["loose"]),
        "p_one_sided_earlier": one_sided(float(mf.params["loose"]), float(mf.pvalues["loose"]), "-"),
        "model": "MixedLM median_relpos_top10_forks ~ loose + C(domain) + (1|prompt)",
    }
    return out
