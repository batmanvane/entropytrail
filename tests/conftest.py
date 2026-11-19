import glob
import json
import os

os.environ.setdefault("HF_HOME", "/Volumes/Models/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
import pytest

REPO = "Qwen/Qwen3.5-9B"


def _real_config() -> dict:
    hf = os.environ["HF_HOME"]
    pat = f"{hf}/hub/models--Qwen--Qwen3.5-9B/snapshots/*/config.json"
    return json.load(open(glob.glob(pat)[0]))


@pytest.fixture(scope="session")
def tiny_model():
    from mlx_lm.models import qwen3_5

    c = _real_config()
    t = dict(c["text_config"])
    t.update(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        layer_types=t["layer_types"][:4], num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, vocab_size=1000,
    )
    c["text_config"] = t
    mx.random.seed(0)
    m = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(c))
    mx.eval(m.parameters())
    return m


@pytest.fixture
def fake_io():
    return {"eos_ids": {999}, "token_texts": lambda ids: [f"<{i}>" for i in ids]}
