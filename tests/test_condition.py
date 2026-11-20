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
    ps = load_prompts(Path("prompts/prompts-v0-smoke.yaml"))
    assert ps.prompts[0].id == "smoke-tool-01" and "sys-v1" in ps.system_prompts


def test_frozen_grid_v1():
    ps = load_prompts(Path("prompts/prompts.yaml"))
    assert ps.version == "v1" and len(ps.prompts) == 21
    cells = {(p.domain, p.openness) for p in ps.prompts}
    assert cells == {(d, o) for d in ("education", "game", "tool") for o in ("tight", "loose")} | {("free", "open")}


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


def test_logits_precision_in_condition_id():
    c = load_condition(COND)
    assert c.logits_precision == "float32"
    assert dataclasses.replace(c, logits_precision="native").condition_id != c.condition_id


def test_load_model_rejects_unknown_logits_precision(monkeypatch):
    import entropytrail.model as m
    import mlx_lm

    monkeypatch.setattr(mlx_lm, "load", lambda repo: (object(), object()))
    c = dataclasses.replace(load_condition(COND), precision="other", logits_precision="fp8")
    with pytest.raises(ValueError):
        m.load_model(c)


def test_golden_condition_id():
    assert load_condition(COND).condition_id == "qwen35-9b-bf16-mlx-1195d88b"


def test_load_model_rejects_runtime_version_mismatch(monkeypatch):
    import entropytrail.model as m
    import mlx_lm

    called = []
    monkeypatch.setattr(mlx_lm, "load", lambda repo: called.append(repo))
    monkeypatch.setattr(m.md, "version", lambda pkg: "0.0.1" if pkg == "mlx-lm" else "1")
    with pytest.raises(RuntimeError, match="mlx-lm"):
        m.load_model(load_condition(COND))
    assert called == []
