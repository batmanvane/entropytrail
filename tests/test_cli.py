import json

import pytest

import entropytrail.cli as cli
import entropytrail.model as model
from entropytrail.condition import Condition, Prompt, PromptSet

GEN = ["generate", "--condition", "c.yaml", "--prompts", "p.yaml",
       "--prompt-id", "p0", "--sample", "0"]
WIN = ["run-window", "--condition", "c.yaml", "--prompts", "p.yaml",
       "--samples", "1", "--until", "06:00"]
RES = ["rescore", "--condition", "c.yaml", "--prompts", "p.yaml"]


class Loaded(Exception):
    pass


def _dirty(monkeypatch, dirty):
    real = model.provenance
    monkeypatch.setattr(model, "provenance", lambda: {**real(), "git_dirty": dirty})

    def no_load(*a, **k):
        raise Loaded()

    monkeypatch.setattr(cli, "_load", no_load)


@pytest.mark.parametrize("argv", [GEN, WIN, RES])
def test_refuses_dirty_tree(monkeypatch, capsys, argv):
    _dirty(monkeypatch, True)
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code != 0
    assert "--allow-dirty" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [GEN, WIN, RES])
def test_allow_dirty_proceeds(monkeypatch, argv):
    _dirty(monkeypatch, True)
    with pytest.raises(Loaded):
        cli.main(argv + ["--allow-dirty"])


@pytest.mark.parametrize("argv", [GEN, WIN, RES])
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


def test_run_window_fork_plan_parser():
    base = ["run-window", "--condition", "c.yaml", "--prompts", "p.yaml", "--until", "06:00"]
    a = cli.build_parser().parse_args(base + ["--fork-plan", "plan.json"])
    assert str(a.fork_plan) == "plan.json" and a.samples is None
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(base + ["--fork-plan", "plan.json", "--samples", "2"])
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(base)


def test_forks_plan_design_options():
    a = cli.build_parser().parse_args(["forks-plan", "--root", "/x", "--prompt-ids", "a", "--apps-per-prompt", "2",
                                       "--seed", "1", "--out", "p.json", "--n-top", "5", "--n-random", "2",
                                       "--n-same", "6"])
    assert (a.n_top, a.n_random, a.n_same, a.extend) == (5, 2, 6, None)
    b = cli.build_parser().parse_args(["forks-plan", "--root", "/x", "--extend", "old.json", "--n-same", "6",
                                       "--out", "p.json"])
    assert str(b.extend) == "old.json" and b.prompt_ids is None


def test_run_window_without_deadline():
    a = cli.build_parser().parse_args(["run-window", "--condition", "c.yaml", "--prompts", "p.yaml", "--samples", "2"])
    assert a.until is None


def test_rescore_command_end_to_end(monkeypatch, tmp_path, tiny_model, fake_io, capsys):
    import hashlib

    from entropytrail.model import generate_recorded
    from entropytrail.storage import write_run

    cond = Condition(name="c", model_repo="r", precision="bf16", runtime="mlx-lm", runtime_version="0",
                     temperature=0.5, max_tokens=50, thinking=False, system_prompt_version="sys-v1",
                     machine="m")
    text = "make an app"
    ps = PromptSet(version="v1", system_prompts={"sys-v1": "sys"},
                   prompts=[Prompt(id="p1", domain="tool", openness="tight", text=text)])
    rec = generate_recorded(tiny_model, [1, 2, 3], seed=1, temperature=0.5, max_tokens=30,
                            eos_ids=set(), token_texts=fake_io["token_texts"])
    sha = hashlib.sha256(text.encode()).hexdigest()
    write_run(tmp_path / cond.condition_id / "p1" / "000", rec, {"prompt_text_sha256": sha})
    write_run(tmp_path / cond.condition_id / "p1" / "001", rec, {"prompt_text_sha256": "other"})
    monkeypatch.setenv("ENTROPYTRAIL_DATA", str(tmp_path))
    monkeypatch.setattr(model, "provenance", lambda: {"git_commit": "x", "git_dirty": False})
    monkeypatch.setattr(cli, "_load", lambda a: (cond, ps, tiny_model, None, "fp"))
    monkeypatch.setattr(model, "build_prompt_ids", lambda tok, s, u, t: [1, 2, 3])
    monkeypatch.setattr(model, "tokenizer_io", lambda tok: ({999}, None))
    with pytest.raises(SystemExit) as e:
        cli.main(RES)
    assert e.value.code == 1  # the unit with the wrong prompt hash fails, the other is rescored
    last = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert last == {"done": 1, "skipped": 0, "failed": 1}
    meta = json.loads((tmp_path / cond.condition_id / "p1" / "000" / "rescore.json").read_text())
    assert meta["temperature"] == 0.5 and meta["max_abs_dH"] < 1e-3 and meta["eos_ids"] == [999]
