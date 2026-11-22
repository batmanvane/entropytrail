import json

import pytest

import entropytrail.cli as cli
import entropytrail.model as model
from entropytrail.condition import Condition, Prompt, PromptSet

GEN = ["generate", "--condition", "c.yaml", "--prompts", "p.yaml",
       "--prompt-id", "p0", "--sample", "0"]
WIN = ["run-window", "--condition", "c.yaml", "--prompts", "p.yaml",
       "--samples", "1", "--until", "06:00"]


class Loaded(Exception):
    pass


def _dirty(monkeypatch, dirty):
    real = model.provenance
    monkeypatch.setattr(model, "provenance", lambda: {**real(), "git_dirty": dirty})

    def no_load(*a, **k):
        raise Loaded()

    monkeypatch.setattr(cli, "_load", no_load)


@pytest.mark.parametrize("argv", [GEN, WIN])
def test_refuses_dirty_tree(monkeypatch, capsys, argv):
    _dirty(monkeypatch, True)
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code != 0
    assert "--allow-dirty" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [GEN, WIN])
def test_allow_dirty_proceeds(monkeypatch, argv):
    _dirty(monkeypatch, True)
    with pytest.raises(Loaded):
        cli.main(argv + ["--allow-dirty"])


@pytest.mark.parametrize("argv", [GEN, WIN])
def test_clean_tree_proceeds(monkeypatch, argv):
    _dirty(monkeypatch, False)
    with pytest.raises(Loaded):
        cli.main(argv)


def test_validation_json_has_provenance(monkeypatch, tmp_path, tiny_model):
    cond = Condition(
        name="t", model_repo="x/y", precision="bf16", runtime="mlx", runtime_version="0",
        temperature=1.0, max_tokens=20, thinking=False, system_prompt_version="v1",
        machine="test",
    )
    ps = PromptSet(version="p1", system_prompts={"v1": "sys"},
                   prompts=[Prompt(id="p0", domain="d", openness="o", text="t0")])
    monkeypatch.setenv("ENTROPYTRAIL_DATA", str(tmp_path))
    monkeypatch.setattr(cli, "_load", lambda args: (cond, ps, tiny_model, None, "fp-test"))
    monkeypatch.setattr(model, "build_prompt_ids", lambda *a: [1, 2, 3])
    monkeypatch.setattr(model, "tokenizer_io",
                        lambda tok: ({999}, lambda ids: [f"<{i}>" for i in ids]))
    args = cli.build_parser().parse_args(
        ["validate", "--condition", "c.yaml", "--prompts", "p.yaml", "--tokens", "20"])
    cli.cmd_validate(args)
    (path,) = tmp_path.glob(f"_validation/{cond.condition_id}/*/validation.json")
    v = json.loads(path.read_text())
    assert v["weights_fingerprint"] == "fp-test"
    for k in ("git_commit", "git_dirty", "mlx", "mlx_lm", "python", "macos", "hostname"):
        assert k in v


def test_eval_parser():
    a = cli.build_parser().parse_args(["eval", "--root", "/x", "--prompt-ids", "a", "b", "--forks"])
    assert a.cmd == "eval" and str(a.root) == "/x" and a.prompt_ids == ["a", "b"] and a.forks
    a = cli.build_parser().parse_args(["eval", "--root", "/x"])
    assert a.prompt_ids is None and not a.forks


def test_forks_plan_parser():
    a = cli.build_parser().parse_args(["forks-plan", "--root", "/x", "--prompt-ids", "a", "b",
                                       "--apps-per-prompt", "5", "--seed", "2026", "--out", "p.json"])
    assert a.cmd == "forks-plan" and a.apps_per_prompt == 5 and a.seed == 2026 and a.prompt_ids == ["a", "b"]
