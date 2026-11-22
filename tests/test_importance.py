import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from entropytrail.importance import fork_importance
from entropytrail.outcome import Outcome

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("collect_forks", ROOT / "scripts" / "collect_forks.py")
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)


def _o(r, v):
    v = np.asarray(v, np.float32)
    return Outcome(r, v / np.linalg.norm(v))


def test_importance_zero_when_alts_equal_sames():
    s = _o(True, [1, 0])
    r = fork_importance([s, s, s], [s, s])
    assert r["importance"] == 0.0 and r["n_alt"] == 3 and r["n_same"] == 2


def test_importance_positive_when_alt_breaks_app():
    s, broken = _o(True, [1, 0]), _o(False, [1, 0])
    r = fork_importance([broken], [s, s])
    assert r["alt_same"] == 1.0 and r["noise_floor"] == 0.0 and r["importance"] == 1.0


def test_noise_floor_subtracted_and_original_secondary():
    a, b, c = _o(True, [1, 0]), _o(True, [0, 1]), _o(True, [1, 1])
    r = fork_importance([b], [a, a], original=b)
    assert r["importance"] == pytest.approx(1.0)
    # with the original (= b) as third same: alt_same = mean(1,1,0) = 2/3, floor = mean(0,1,1) = 2/3
    assert r["importance_incl_original"] == pytest.approx(0.0)
    assert fork_importance([c], [a, a])["importance_incl_original"] is None


def test_too_few_sames_gives_none():
    s = _o(True, [1, 0])
    assert fork_importance([s], [s])["importance"] is None
    assert fork_importance([], [s, s])["importance"] is None


def _eval(u, renders, screen):
    (u / "eval").mkdir(parents=True)
    v = np.asarray(screen, np.float32)
    v /= np.linalg.norm(v)
    np.savez(u / "eval" / "embeddings.npz", screen=v)
    (u / "eval" / "render.json").write_text(json.dumps({"renders": renders}))
    (u / "eval" / "eval.json").write_text(json.dumps({"status": "done"}))


def test_collect_rows(tmp_path):
    cond = tmp_path / "c"
    parent = cond / "p1" / "000"
    _eval(parent, True, [1, 0])
    (parent / "meta.json").write_text(json.dumps({"prompt": {"domain": "game", "openness": "tight"}}))
    for v, (r, s) in {"alt1": (False, [1, 0]), "alt2": (True, [1, 0]), "same1": (True, [1, 0]),
                      "same2": (True, [1, 0])}.items():
        _eval(parent / "forks" / "00012" / v, r, s)
    plan = {"apps": [{"prompt_id": "p1", "sample": 0, "sites": [
        {"pos": 12, "kind": "top", "H": 3.0, "S": 1.0, "section": "js", "rel_pos": 0.2,
         "variants": {"alt1": 5, "alt2": 6, "alt3": 7, "same1": 1, "same2": 1}},
        {"pos": 40, "kind": "random", "H": 0.1, "S": 0.0, "section": "css", "rel_pos": 0.5,
         "variants": {"alt1": 5, "same1": 1, "same2": 1}}]}]}
    rows = cf.collect(cond, plan)
    assert len(rows) == 1  # the second site has no evaluated rollouts yet
    r = rows[0]
    assert r["pos"] == 12 and r["n_alt"] == 2 and r["n_alt_renders"] == 1 and r["n_same_renders"] == 2
    assert r["importance"] == pytest.approx(0.5) and r["domain"] == "game"
