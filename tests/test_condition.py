import dataclasses
from pathlib import Path

import pytest
import yaml

from entropytrail.condition import derive_seed, load_condition, load_prompts

COND = Path("configs/conditions/qwen35-9b-bf16-mlx.yaml")


def test_condition_id_stable_and_sensitive():
    c = load_condition(COND)
    assert c.condition_id == load_condition(COND).condition_id
    assert c.condition_id.startswith("qwen35-9b-bf16-mlx-")
    assert dataclasses.replace(c, temperature=0.7).condition_id != c.condition_id


def test_prompts_load():
    ps = load_prompts(Path("prompts/prompts.yaml"))
    assert ps.prompts[0].id == "smoke-tool-01" and "sys-v1" in ps.system_prompts


def test_prompt_ids_unique(tmp_path):
    p = tmp_path / "p.yaml"
    entry = {"id": "a", "domain": "tool", "openness": "tight", "text": "x"}
    p.write_text(
        yaml.safe_dump({"version": "v", "system_prompts": {"s": "t"}, "prompts": [entry, dict(entry)]})
    )
    with pytest.raises(ValueError):
        load_prompts(p)


def test_seed_deterministic_and_distinct():
    assert derive_seed("c", "p", 0) == derive_seed("c", "p", 0)
    assert len({derive_seed("c", "p", i) for i in range(100)}) == 100
    assert 0 <= derive_seed("c", "p", 0) < 2**32
