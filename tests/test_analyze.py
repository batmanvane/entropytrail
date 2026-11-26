import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("statsmodels")
import numpy as np
import pandas as pd

from test_hypotheses import _apps, _forks, grid_prompts

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analyze", ROOT / "scripts" / "analyze.py")
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)


def inputs():
    rng = np.random.default_rng(0)
    ens = grid_prompts()
    ens["ensemble_spread"] = 0.3 + 0.1 * (ens["openness"] == "loose") + rng.normal(0, 0.02, 21)
    meas = ens.loc[ens.index.repeat(4), ["prompt_id", "domain", "openness"]].reset_index(drop=True)
    meas["sample"] = list(range(4)) * 21
    for c in ("median_relpos_top10_forks", "late_js_surprise", "fork_surprise", "path_action_per_token"):
        meas[c] = rng.random(len(meas))
    meas["finish_reason"] = "eos"
    meas.loc[0, "finish_reason"] = "max_tokens"
    frames = dict(list(_apps("bimodal", n_apps=len(meas)).items()))
    frames = {(p, s): f for (p, s), f in zip(zip(meas["prompt_id"], meas["sample"]), frames.values())}
    return meas, ens, frames


def test_summary_without_forks_and_ratings_is_interim():
    meas, ens, frames = inputs()
    s = an.build_summary(meas, ens, frames, importance=None, human=None)
    assert s["H1"]["status"] == "done" and s["H2"]["status"] == "done"
    assert s["H3"]["status"].startswith("waiting") and s["H4"]["status"].startswith("waiting")
    assert s["interim"] is True and set(s["holm"]) == {"H1", "H2"}
    assert "sensitivity" in s["H1"] and "sensitivity" in s["H2"]


def test_summary_with_forks_includes_h3_and_h_ge_2_sensitivity():
    meas, ens, frames = inputs()
    s = an.build_summary(meas, ens, frames, importance=_forks("section"), human=None)
    assert s["H3"]["status"] == "done" and "H_ge_2" in s["H3"]["sensitivity"]
    assert set(s["holm"]) == {"H1", "H2", "H3"}


def test_guard_requires_d036(tmp_path):
    (tmp_path / "DECISIONS.md").write_text("## D-035 · x\n")
    with pytest.raises(SystemExit):
        an.require_readings(tmp_path / "DECISIONS.md")
    (tmp_path / "DECISIONS.md").write_text("## D-036 · 2026-11-23 · readings\n")
    an.require_readings(tmp_path / "DECISIONS.md")
