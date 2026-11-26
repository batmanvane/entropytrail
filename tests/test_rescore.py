import json
import math

import mlx.core as mx
import numpy as np

from entropytrail.model import generate_recorded
from entropytrail.rescore import _stop_stats, grid_units, is_rescored, rescore, rescore_unit
from entropytrail.storage import write_run


def test_stop_stats_by_hand():
    p = np.array([[0.5, 0.25, 0.125, 0.125]])
    lp = mx.array(np.log(p), dtype=mx.float32)
    st = {k: np.array(v) for k, v in _stop_stats(lp, mx.array([2, 3]), 1.0).items()}
    assert math.isclose(st["H"][0], 1.75, abs_tol=1e-5)
    assert math.isclose(st["logp2_eos"][0], -2.0, abs_tol=1e-5)  # 0.125 + 0.125
    np.testing.assert_allclose(st["H_T"], st["H"], atol=1e-6)
    cold = {k: np.array(v) for k, v in _stop_stats(lp, mx.array([2, 3]), 0.5).items()}
    pt = p[0] ** 2 / (p[0] ** 2).sum()  # T = 0.5 squares the probabilities
    assert math.isclose(cold["logp2_eos_T"][0], math.log2(pt[2] + pt[3]), abs_tol=1e-5)
    assert math.isclose(cold["H_T"][0], float(-(pt * np.log2(pt)).sum()), abs_tol=1e-5)
    assert cold["H_T"][0] < st["H"][0]


def test_rescore_matches_recorder_and_writes_unit(tiny_model, fake_io, tmp_path):
    rec = generate_recorded(tiny_model, [1, 2, 3], seed=5, temperature=1.0, max_tokens=80,
                            eos_ids=set(), token_texts=fake_io["token_texts"])
    ids = [t.token_id for t in rec.tokens]
    out = rescore(tiny_model, [1, 2, 3], ids, {7, 999}, 0.7, chunk=16)
    np.testing.assert_allclose(out["H"], [t.stats.H for t in rec.tokens], atol=1e-3)
    assert len(out["logp2_eos"]) == 80 and (out["logp2_eos"] <= 0).all()

    unit = tmp_path / "cond" / "p1" / "000"
    write_run(unit, rec, {"condition_id": "cond"})
    assert grid_units(tmp_path / "cond") == [unit] and not is_rescored(unit)
    res = rescore_unit(unit, tiny_model, [1, 2, 3], {7, 999}, 0.7, {"git_commit": "x"}, chunk=16)
    assert is_rescored(unit) and res["max_abs_dH"] < 1e-3 and res["n_tokens"] == 80
    assert json.loads((unit / "rescore.json").read_text())["temperature"] == 0.7
    import pyarrow.parquet as pq
    t = pq.read_table(unit / "rescore.parquet").to_pandas()
    assert list(t.columns) == ["pos", "H_T", "logp2_eos", "logp2_eos_T"] and len(t) == 80
    assert grid_units(tmp_path / "cond", ["other"]) == []
