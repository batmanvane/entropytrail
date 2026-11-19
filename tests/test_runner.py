import dataclasses
import hashlib
import json
from datetime import datetime, timedelta

import pytest

from entropytrail.condition import Condition, Prompt, PromptSet, derive_seed
from entropytrail.runner import IdentityMismatch, Unit, plan_units, run_units
from entropytrail.storage import is_done, run_dir

BASE = datetime(2026, 1, 1, 23, 0, 0)


def make_cond(max_tokens=20):
    return Condition(
        name="t", model_repo="x/y", precision="bf16", runtime="mlx",
        runtime_version="0", temperature=1.0, max_tokens=max_tokens,
        thinking=False, system_prompt_version="v1", machine="test",
    )


def make_ps():
    return PromptSet(
        version="p1", system_prompts={"v1": "sys"},
        prompts=[Prompt(id=f"p{i}", domain="d", openness="o", text=f"t{i}") for i in range(3)],
    )


class Clock:
    def __init__(self, step=1.0):
        self.n = 0
        self.step = step

    def __call__(self):
        self.n += 1
        return BASE + timedelta(seconds=self.n * self.step)


def go(tiny_model, fake_io, tmp_path, units, cond=None, deadline=None, now=None,
       ps=None, fingerprint="test"):
    cond = cond or make_cond()
    return run_units(
        units, cond, ps or make_ps(), tiny_model, None, tmp_path, deadline,
        fingerprint=fingerprint, now=now or Clock(0.0001),
        io=(fake_io["eos_ids"], fake_io["token_texts"]),
        prompt_ids_fn=lambda p: [1, 2, 3],
    )


def test_plan_units_prompt_major():
    us = plan_units(make_ps(), 2, ["p2", "p0"])
    assert [(u.prompt.id, u.sample) for u in us] == [("p2", 0), ("p2", 1), ("p0", 0), ("p0", 1)]
    assert len(plan_units(make_ps(), 1, None)) == 3


def test_resume_skips_done(tiny_model, fake_io, tmp_path):
    units = plan_units(make_ps(), 1, None)
    r1 = go(tiny_model, fake_io, tmp_path, units)
    assert r1["done"] == 3
    r2 = go(tiny_model, fake_io, tmp_path, units)
    assert r2["done"] == 0 and r2["skipped"] == 3


def test_deadline_mid_generation_discards_partial(tiny_model, fake_io, tmp_path):
    cond = make_cond(max_tokens=50)
    units = plan_units(make_ps(), 1, None)[:1]
    clock = Clock(1.0)
    r = go(tiny_model, fake_io, tmp_path, units, cond=cond,
           deadline=BASE + timedelta(seconds=10), now=clock)
    assert r["aborted"] == 1 and r["done"] == 0
    final = run_dir(tmp_path, cond.condition_id, "p0", 0)
    assert not is_done(final)
    assert not list(tmp_path.rglob("*.partial"))
    r2 = go(tiny_model, fake_io, tmp_path, units, cond=cond)
    assert r2["done"] == 1 and is_done(final)


def test_deadline_before_unit_not_aborted(tiny_model, fake_io, tmp_path):
    units = plan_units(make_ps(), 1, None)
    r = go(tiny_model, fake_io, tmp_path, units, deadline=BASE)
    assert r["done"] == 0 and r["aborted"] == 0


def test_stale_partial_cleaned(tiny_model, fake_io, tmp_path):
    cond = make_cond()
    final = run_dir(tmp_path, cond.condition_id, "p0", 0)
    stale = final.parent / "000.partial"
    stale.mkdir(parents=True)
    (stale / "junk").write_text("x")
    r = go(tiny_model, fake_io, tmp_path, plan_units(make_ps(), 1, ["p0"]), cond=cond)
    assert not stale.exists()
    assert r["done"] == 1 and is_done(final)


def test_runlog_appended(tiny_model, fake_io, tmp_path):
    units = plan_units(make_ps(), 1, ["p0"])
    r1 = go(tiny_model, fake_io, tmp_path, units)
    go(tiny_model, fake_io, tmp_path, units)
    lines = (tmp_path / "runlog.jsonl").read_text().splitlines()
    assert len(lines) == 2
    d = json.loads(lines[0])
    assert set(d) == {
        "started", "ended", "condition_id", "done", "skipped", "aborted",
        "error", "git_commit", "weights_fingerprint", "stale_moved",
    }
    assert d == r1
    assert d["error"] is None and d["stale_moved"] == []
    assert d["weights_fingerprint"] == "test" and d["git_commit"]


def test_meta_has_provenance(tiny_model, fake_io, tmp_path):
    cond = make_cond()
    go(tiny_model, fake_io, tmp_path, plan_units(make_ps(), 1, ["p1"]), cond=cond)
    meta = json.loads((run_dir(tmp_path, cond.condition_id, "p1", 0) / "meta.json").read_text())
    assert meta["seed"] == derive_seed(cond.condition_id, "p1", 0)
    assert meta["condition_id"] == cond.condition_id
    assert meta["weights_fingerprint"] == "test"
    assert meta["git_commit"]
    assert meta["prompt"]["id"] == "p1" and meta["condition"]["name"] == "t"
    assert meta["prompt_set_version"] == "p1" and meta["sample"] == 0
    assert meta["system_prompt_version"] == "v1"
    assert meta["started_at"] and meta["finished_at"]
    assert meta["prompt_ids"] == [1, 2, 3]
    assert meta["prompt_text_sha256"] == hashlib.sha256(b"t1").hexdigest()
    assert meta["system_prompt_sha256"] == hashlib.sha256(b"sys").hexdigest()


def _ps_with(system="sys", texts=("t0", "t1", "t2")):
    return PromptSet(
        version="p1", system_prompts={"v1": system},
        prompts=[Prompt(id=f"p{i}", domain="d", openness="o", text=t) for i, t in enumerate(texts)],
    )


def _last_log(tmp_path):
    return json.loads((tmp_path / "runlog.jsonl").read_text().splitlines()[-1])


def test_condition_manifest_created(tiny_model, fake_io, tmp_path):
    cond = make_cond()
    go(tiny_model, fake_io, tmp_path, plan_units(make_ps(), 1, ["p0"]), cond=cond)
    m = json.loads((tmp_path / cond.condition_id / "condition.json").read_text())
    assert m["condition_id"] == cond.condition_id
    assert m["condition"] == dataclasses.asdict(cond)
    assert m["weights_fingerprint"] == "test"
    assert m["system_prompt_version"] == "v1"
    assert m["system_prompt_sha256"] == hashlib.sha256(b"sys").hexdigest()
    assert m["mlx"] and m["mlx_lm"] and m["created_at"]


@pytest.mark.parametrize("change", ["fingerprint", "system_prompt", "mlx", "mlx_lm"])
def test_manifest_mismatch_aborts(tiny_model, fake_io, tmp_path, change, monkeypatch):
    cond = make_cond()
    units = plan_units(make_ps(), 1, None)
    go(tiny_model, fake_io, tmp_path, units[:1], cond=cond)
    kw = {}
    if change == "fingerprint":
        kw["fingerprint"] = "other"
    elif change == "system_prompt":
        kw["ps"] = _ps_with(system="changed")
    else:
        import entropytrail.runner as runner

        real = runner.provenance
        monkeypatch.setattr(runner, "provenance", lambda: {**real(), change: "9.9.9"})
    with pytest.raises(IdentityMismatch, match="condition.json"):
        go(tiny_model, fake_io, tmp_path, units, cond=cond, **kw)
    assert not is_done(run_dir(tmp_path, cond.condition_id, "p1", 0))
    log = _last_log(tmp_path)
    assert log["done"] == 0 and "IdentityMismatch" in log["error"]


def test_prompt_text_change_on_done_unit_aborts(tiny_model, fake_io, tmp_path):
    cond = make_cond()
    units = plan_units(make_ps(), 1, ["p0"])
    go(tiny_model, fake_io, tmp_path, units, cond=cond)
    ps2 = _ps_with(texts=("CHANGED", "t1", "t2"))
    with pytest.raises(IdentityMismatch, match="p0"):
        go(tiny_model, fake_io, tmp_path, plan_units(ps2, 1, ["p0"]), cond=cond, ps=ps2)
    assert "IdentityMismatch" in _last_log(tmp_path)["error"]


def test_crash_writes_runlog_and_reraises(tiny_model, fake_io, tmp_path):
    cond = make_cond()

    def boom(p):
        raise RuntimeError("kaputt")

    with pytest.raises(RuntimeError, match="kaputt"):
        run_units(
            plan_units(make_ps(), 1, ["p0"]), cond, make_ps(), tiny_model, None, tmp_path,
            None, fingerprint="test", now=Clock(0.0001),
            io=(fake_io["eos_ids"], fake_io["token_texts"]), prompt_ids_fn=boom,
        )
    log = _last_log(tmp_path)
    assert log["error"] == "RuntimeError: kaputt"
    assert log["done"] == 0 and log["weights_fingerprint"] == "test"


def test_stale_final_dir_moved_and_redone(tiny_model, fake_io, tmp_path):
    cond = make_cond()
    final = run_dir(tmp_path, cond.condition_id, "p0", 0)
    final.mkdir(parents=True)
    (final / "meta.json").write_text('{"status": "writing"}')
    r = go(tiny_model, fake_io, tmp_path, plan_units(make_ps(), 1, ["p0"]), cond=cond)
    assert r["done"] == 1 and is_done(final)
    assert len(r["stale_moved"]) == 1
    moved = tmp_path / r["stale_moved"][0]
    assert moved.name.startswith("000.stale-") and (moved / "meta.json").exists()


def test_cli_help_parses():
    from entropytrail.cli import build_parser

    p = build_parser()
    for cmd in ["generate", "run-window", "validate", "bench"]:
        with pytest.raises(SystemExit) as e:
            p.parse_args([cmd, "--help"])
        assert e.value.code == 0
    from entropytrail.cli import next_occurrence

    assert next_occurrence("06:00", datetime(2026, 1, 1, 23, 0)) == datetime(2026, 1, 2, 6, 0)
    assert next_occurrence("06:00", datetime(2026, 1, 1, 5, 0)) == datetime(2026, 1, 1, 6, 0)
    assert next_occurrence("06:00", datetime(2026, 1, 1, 6, 0)) == datetime(2026, 1, 2, 6, 0)
