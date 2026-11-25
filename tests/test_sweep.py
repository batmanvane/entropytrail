import numpy as np
import pandas as pd
import pytest

from entropytrail.analysis.sweep import c1_renders, c2_spread, c3_novelty, compare, stack

DOMAINS = {"e": "education", "g": "game", "t": "tool"}


def _apps(seed, p_render, nov_shift, n_prompts=12, n=20):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_prompts):
        dom = list(DOMAINS.values())[i % 3]
        op = "loose" if i % 2 else "tight"
        base = rng.normal(0, 0.5)
        for s in range(n):
            rows.append({"prompt_id": f"p{i:02d}", "sample": s, "domain": dom, "openness": op,
                         "renders": bool(rng.random() < p_render),
                         "novelty_calibrated": float(np.clip(round(2 + base + nov_shift + rng.normal(0, 0.8)), 1, 5))})
    return pd.DataFrame(rows)


def _ens(seed, shift, n_prompts=12):
    rng = np.random.default_rng(seed)
    return pd.DataFrame([{"prompt_id": f"p{i:02d}", "domain": list(DOMAINS.values())[i % 3],
                          "openness": "loose" if i % 2 else "tight",
                          "ensemble_spread": 0.15 + 0.01 * i + shift + rng.normal(0, 0.005)} for i in range(n_prompts)])


def test_stack_refuses_duplicate_units_and_mixed_conditions():
    a = _apps(0, 0.7, 0)
    with pytest.raises(ValueError):
        stack({1.0: a, 0.7: pd.concat([a, a.head(3)])})
    mixed = a.assign(condition_id=["x"] * 10 + ["y"] * (len(a) - 10))
    with pytest.raises(ValueError):
        stack({1.0: mixed, 0.7: a})
    s = stack({1.0: a, 0.7: a})
    assert set(s["temperature"]) == {1.0, 0.7} and len(s) == 2 * len(a)


def test_c1_more_renders_at_low_t():
    d = stack({1.0: _apps(1, 0.6, 0), 0.7: _apps(2, 0.9, 0)})
    r = c1_renders(d, low=0.7)
    assert r["estimate"] > 0 and r["p_primary"] < 0.01 and r["sign"] == "+"


def test_c2_lower_spread_at_low_t_and_wrong_direction():
    d = stack({1.0: _ens(1, 0.0), 0.7: _ens(2, -0.03)})
    r = c2_spread(d, low=0.7)
    assert r["estimate"] < 0 and r["p_primary"] < 0.01 and r["n"] == 24
    up = c2_spread(stack({1.0: _ens(1, 0.0), 0.7: _ens(2, +0.03)}), low=0.7)
    assert up["p_primary"] > 0.5


def test_c3_drops_null_novelty():
    lo = _apps(4, 0.7, -0.6)
    lo.loc[:9, "novelty_calibrated"] = np.nan
    d = stack({1.0: _apps(3, 0.7, 0), 0.7: lo})
    r = c3_novelty(d, low=0.7)
    assert r["estimate"] < 0 and r["p_primary"] < 0.05 and r["n"] == 2 * 240 - 10


def test_compare_holm_family_of_three():
    out = compare(stack({1.0: _apps(1, 0.6, 0), 0.7: _apps(2, 0.9, -0.6)}),
                  stack({1.0: _ens(1, 0.0), 0.7: _ens(2, -0.03)}), low=0.7)
    assert set(out["holm"]) == {"C1", "C2", "C3"}
    assert all(v["reject"] for v in out["holm"].values())
    p = sorted(v["p"] for v in out["holm"].values())
    assert out["holm"][min(out["holm"], key=lambda k: out["holm"][k]["p"])]["p_holm"] == pytest.approx(min(1, 3 * p[0]))
