import numpy as np
import pytest

from entropytrail.outcome import (
    Outcome, cosine_distances, ensemble_rarity, ensemble_spread, outcome_distance,
    outcome_distances, rarity, screen_embedding, spread,
)


def _o(r, v):
    v = np.asarray(v, np.float32)
    return Outcome(r, v / np.linalg.norm(v))


def test_render_mismatch_is_max_distance():
    assert outcome_distance(_o(True, [1, 0]), _o(False, [1, 0])) == 1.0


def test_same_screen_zero_and_orthogonal_one():
    assert outcome_distance(_o(True, [1, 0]), _o(True, [1, 0])) == 0.0
    assert outcome_distance(_o(False, [1, 0]), _o(False, [0, 1])) == pytest.approx(1.0)


def test_spread_and_rarity():
    os_ = [_o(True, [1, 0]), _o(True, [1, 0]), _o(False, [0, 1])]
    assert ensemble_spread(os_) == pytest.approx(2 / 3)
    assert ensemble_rarity(os_).tolist() == pytest.approx([0.5, 0.5, 1.0])
    assert ensemble_spread(os_, use_render_rule=True) == pytest.approx(2 / 3)


def test_render_rule_changes_spread():
    os_ = [_o(True, [1, 0]), _o(False, [1, 0])]
    assert ensemble_spread(os_) == 0.0 and ensemble_spread(os_, use_render_rule=True) == 1.0
    assert outcome_distances(os_)[0, 1] == 1.0


def test_generic_matrix_helpers():
    E = np.array([[1, 0], [0, 1], [1, 0]], np.float32)
    D = cosine_distances(E)
    assert D.shape == (3, 3) and np.all(np.diag(D) == 0)
    assert spread(D) == pytest.approx(2 / 3) and rarity(D).tolist() == pytest.approx([0.5, 1.0, 0.5])


def test_screen_embedding_normalised():
    e = screen_embedding(np.array([1, 0], np.float32), np.array([0, 1], np.float32))
    assert np.linalg.norm(e) == pytest.approx(1.0)
