import mlx.core as mx
import numpy as np
import pytest

from entropytrail.stats import TOPK, token_stats, token_stats_batch


def lp(p):
    return mx.log(mx.array(p, dtype=mx.float32))


def test_uniform_four():
    s = token_stats(lp([.25, .25, .25, .25]), 2)
    assert s.H == pytest.approx(2.0, abs=1e-5)
    assert s.S == pytest.approx(2.0, abs=1e-5)
    assert s.varentropy == pytest.approx(0.0, abs=1e-5)
    assert s.margin == pytest.approx(0.0, abs=1e-5)


def test_skewed():
    s = token_stats(lp([.5, .25, .125, .125]), 3)
    assert s.H == pytest.approx(1.75, abs=1e-5)
    assert s.S == pytest.approx(3.0, abs=1e-5)
    assert s.varentropy == pytest.approx(0.6875, abs=1e-5)  # E[(-log2p)^2] - H^2 = 3.75 - 3.0625
    assert s.margin == pytest.approx(1.0, abs=1e-5)


def test_topk_sorted_and_padded_shape():
    v = mx.log(mx.softmax(mx.random.normal((1000,), key=mx.random.key(0))))
    s = token_stats(v, 5)
    assert s.topk_ids.shape == (TOPK,) and np.all(np.diff(s.topk_logp2) <= 0)
    assert s.topk_ids[0] == int(mx.argmax(v))
    assert s.topk_ids.dtype == np.int32 and s.topk_logp2.dtype == np.float32


def test_small_vocab_topk_is_min():
    s = token_stats(lp([.5, .25, .125, .125]), 0)
    assert s.topk_ids.shape == (4,)


def test_zero_probability_entries_do_not_nan():
    s = token_stats(mx.array([0.0, -mx.inf, -mx.inf, -mx.inf]), 0)
    assert s.H == 0.0 and s.S == 0.0
    assert s.varentropy == 0.0


def test_bf16_input_is_cast():
    v = lp([.5, .25, .125, .125]).astype(mx.bfloat16)
    s = token_stats(v, 0)
    assert s.H == pytest.approx(1.75, abs=1e-2)


def test_batch_matches_single():
    v = mx.log(mx.softmax(mx.random.normal((7, 500), key=mx.random.key(1)), axis=-1))
    ids = mx.array([3, 1, 4, 1, 5, 9, 2])
    b = token_stats_batch(v, ids)
    assert b["topk_ids"].shape == (7, TOPK) and b["topk_logp2"].shape == (7, TOPK)
    for t in range(7):
        s = token_stats(v[t], int(ids[t]))
        assert b["H"][t] == pytest.approx(s.H, abs=1e-5) and b["S"][t] == pytest.approx(s.S, abs=1e-5)
        assert b["varentropy"][t] == pytest.approx(s.varentropy, abs=1e-5)
        assert b["margin"][t] == pytest.approx(s.margin, abs=1e-5)
        assert np.array_equal(b["topk_ids"][t], s.topk_ids)
