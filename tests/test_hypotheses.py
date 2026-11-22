import pytest

pytest.importorskip("statsmodels")
import numpy as np
import pandas as pd

from entropytrail.analysis.tests_h import design, h1, h2

DOMAINS = ["education", "game", "tool"]


def grid_prompts():
    rows = [{"prompt_id": f"{d[:4]}-{o}-0{k}", "domain": d, "openness": o}
            for d in DOMAINS for o in ("tight", "loose") for k in range(1, 4)]
    rows += [{"prompt_id": f"free-0{k}", "domain": "free", "openness": "open"} for k in range(1, 4)]
    return pd.DataFrame(rows)


def test_design_coding():
    d = design(grid_prompts())
    assert d["loose"].sum() == 9 and set(d["domain"]) == {"education", "game", "tool", "free"}
    assert d.loc[d["domain"] == "free", "loose"].eq(0).all()


def _apps(kind, n_apps=20, n_tok=1500, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n_apps):
        if kind == "bimodal":
            k = int(n_tok * 0.7)
            logh = np.r_[rng.normal(-2.5, 0.2, k), rng.normal(0.2, 0.15, n_tok - k)]
        else:
            logh = rng.normal(-1, 0.5, n_tok)
        out[f"a{i}"] = pd.DataFrame({"H": 10 ** logh, "is_code": rng.random(n_tok) > 0.1})
    return out


def test_h1_bimodal_supported():
    r = h1(_apps("bimodal"))
    assert r["share_rejecting"] >= 0.95 and r["support_frozen"] and r["p_primary"] < 0.05
    assert r["n"] == 20 and len(r["per_app"]) == 20 and "share_rejecting_code_only" in r


def test_h1_unimodal_not_supported():
    r = h1(_apps("unimodal"))
    assert r["share_rejecting"] <= 0.2 and not r["support_frozen"] and r["p_primary"] > 0.5


def test_h2_detects_loose_spread():
    rng = np.random.default_rng(3)
    ens = grid_prompts()
    ens["ensemble_spread"] = 0.3 + 0.2 * (ens["openness"] == "loose") + rng.normal(0, 0.02, len(ens))
    meas = ens.loc[ens.index.repeat(5), ["prompt_id", "domain", "openness"]].reset_index(drop=True)
    meas["median_relpos_top10_forks"] = rng.random(len(meas))
    r = h2(ens, meas)
    assert r["n"] == 21 and r["estimate"] == pytest.approx(0.2, abs=0.03)
    assert r["p_primary"] < 0.05 and r["ci"][0] > 0
    assert "secondary_relpos" in r


def test_h2_one_sided_wrong_direction():
    rng = np.random.default_rng(4)
    ens = grid_prompts()
    ens["ensemble_spread"] = 0.5 - 0.2 * (ens["openness"] == "loose") + rng.normal(0, 0.02, len(ens))
    meas = ens.loc[ens.index.repeat(5), ["prompt_id", "domain", "openness"]].reset_index(drop=True)
    meas["median_relpos_top10_forks"] = rng.random(len(meas))
    assert h2(ens, meas)["p_primary"] > 0.95


def _forks(effect_on="section", seed=5, sections=("css", "js", "text")):
    from entropytrail.analysis.tests_h import h3  # noqa: F401
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(15):
        for s in range(13):
            sec = sections[rng.integers(len(sections))]
            H = rng.uniform(0, 6)
            base = {"css": 0.05, "js": 0.4, "text": 0.2, "html": 0.1}[sec] if effect_on == "section" else 0.08 * H
            rows.append({"prompt_id": "p", "sample": a, "app": f"a{a}", "pos": s * 50, "H": H, "section": sec,
                         "rel_pos": s / 13, "kind": "top" if s < 10 else "random",
                         "importance": base + rng.normal(0, 0.05)})
    return pd.DataFrame(rows)


def test_h3_section_matters_not_entropy():
    from entropytrail.analysis.tests_h import h3
    r = h3(_forks("section"))
    assert r["p_primary"] < 0.01 and r["lrt_df"] == 3  # rel_pos + 2 section dummies
    assert abs(r["spearman_rho"]) < 0.5 and r["rho_ci"][0] < r["spearman_rho"] < r["rho_ci"][1]
    assert set(r["by_kind"]) == {"top", "random"}


def test_h3_df_from_realised_sections():
    from entropytrail.analysis.tests_h import h3
    assert h3(_forks("section", sections=("css", "js")))["lrt_df"] == 2


def test_h3_entropy_only():
    from entropytrail.analysis.tests_h import h3
    r = h3(_forks("H"))
    assert r["spearman_rho"] > 0.5
