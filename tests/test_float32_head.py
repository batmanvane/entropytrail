import mlx.core as mx
import pytest

from conftest import make_tiny_model
from entropytrail.model import _use_float32_head


def test_float32_head_untied():
    m = make_tiny_model(tie_word_embeddings=False)
    _use_float32_head(m)
    assert m.language_model.lm_head.weight.dtype == mx.float32
    assert m(mx.array([[1, 2, 3]])).dtype == mx.float32


def test_float32_head_tied_raises():
    m = make_tiny_model(tie_word_embeddings=True)
    with pytest.raises(ValueError):
        _use_float32_head(m)
