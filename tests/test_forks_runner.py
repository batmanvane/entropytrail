import dataclasses
import json
from datetime import datetime, timedelta

import pytest

from entropytrail.condition import Condition, Prompt, PromptSet
from entropytrail.forks import rollout_dir, run_rollouts
from entropytrail.runner import IdentityMismatch, Unit, run_units
from entropytrail.storage import is_done, read_tokens, run_dir

BASE = datetime(2026, 1, 1, 23, 0, 0)
EOS = 999


def make_cond(max_tokens=20):
    return Condition(
        name="t", model_repo="x/y", precision="bf16", runtime="mlx",
        runtime_version="0", temperature=1.0, max_tokens=max_tokens,
        thinking=False, system_prompt_version="v1", machine="test",
    )


def make_ps(text="t0"):
    return PromptSet(version="p1", system_prompts={"v1": "sys"},
                     prompts=[Prompt(id="p0", domain="d", openness="o", text=text)])


class Clock:
    def __init__(self):
        self.late = False

    def __call__(self):
        return BASE + (timedelta(hours=1) if self.late else timedelta(0))


@pytest.fixture
def parent(tiny_model, fake_io, tmp_path):
    cond, ps = make_cond(), make_ps()
    run_units([Unit(ps.prompts[0], 0)], cond, ps, tiny_model, None, tmp_path, None,
              fingerprint="test", now=Clock(), io=(fake_io["eos_ids"], fake_io["token_texts"]),
              prompt_ids_fn=lambda p: [1, 2, 3])
    u = run_dir(tmp_path, cond.condition_id, "p0", 0)
    return u, read_tokens(u).to_pydict()["token_id"]


def one_site_plan(cond, pos, actual, variants, sample=0):
    return {"plan_version": 1, "condition_id": cond.condition_id, "seed": 0, "params": {},
            "apps": [{"prompt_id": "p0", "sample": sample, "n_tokens": None, "sites": [
                {"pos": pos, "kind": "top", "H": 2.5, "S": 1.0, "section": "js", "rel_pos": 0.1,
                 "actual_id": actual, "variants": variants}]}]}


def go_forks(tiny_model, io, root, plan, cond=None, ps=None, deadline=None, now=None):
    return run_rollouts(plan, cond or make_cond(), ps or make_ps(), tiny_model, None, root, deadline,
                        fingerprint="test", now=now or Clock(), io=io, plan_name="plan-test.json",
                        plan_sha256="abc")


def io_of(fake_io):
    return (fake_io["eos_ids"], fake_io["token_texts"])


def test_rollouts_written_with_full_text(tiny_model, fake_io, tmp_path, parent):
    u, ids = parent
    plan = one_site_plan(make_cond(), 3, ids[3], {"alt1": 7, "same1": ids[3], "same2": ids[3]})
    r = go_forks(tiny_model, io_of(fake_io), tmp_path, plan)
    assert r["done"] == 3 and r["kind"] == "forks"
    d = rollout_dir(u, 3, "alt1")
    assert d == u / "forks" / "00003" / "alt1" and is_done(d)
    meta = json.loads((d / "meta.json").read_text())
    f = meta["fork"]
    assert f["forced_token_id"] == 7 and f["pos"] == 3 and f["actual_token_id"] == ids[3]
    assert f["parent"] == "p0/000" and f["prefix_len"] == 3 + 3 + 1 and meta["prompt_ids"] == [1, 2, 3]
    out = (d / "output.txt").read_text()
    prefix = "".join(f"<{i}>" for i in ids[:3]) + "<7>"
    assert out.startswith(prefix) and f["continuation_char_offset"] == len(prefix)
    cont = read_tokens(d).to_pydict()
    assert out[len(prefix):] == "".join(cont["text"])
    assert len(cont["token_id"]) <= 20 - 3 - 1


def test_deadline_keeps_finished_rollouts_and_resumes(tiny_model, fake_io, tmp_path, parent):
    u, ids = parent
    plan = one_site_plan(make_cond(), 3, ids[3], {"alt1": 7, "alt2": 8, "alt3": 9, "same1": ids[3]})
    clock = Clock()
    real_tt = fake_io["token_texts"]
    calls = {"n": 0}

    def tt(xs):  # the deadline passes while the 3rd rollout is being decoded
        calls["n"] += 1
        return real_tt(xs)

    def now():
        clock.late = sum(is_done(rollout_dir(u, 3, v)) for v in ("alt1", "alt2")) == 2
        return clock()

    r = go_forks(tiny_model, (fake_io["eos_ids"], tt), tmp_path, plan, deadline=BASE + timedelta(minutes=1), now=now)
    assert r["done"] == 2 and not rollout_dir(u, 3, "alt3").exists()
    assert not any(p.name.endswith(".partial") for p in (u / "forks" / "00003").iterdir())
    r2 = go_forks(tiny_model, io_of(fake_io), tmp_path, plan)
    assert r2["skipped"] == 2 and r2["done"] == 2


def test_forced_eos_and_zero_budget(tiny_model, fake_io, tmp_path, parent):
    u, ids = parent
    plan = one_site_plan(make_cond(), 3, ids[3], {"alt1": EOS})
    go_forks(tiny_model, io_of(fake_io), tmp_path, plan)
    m = json.loads((rollout_dir(u, 3, "alt1") / "meta.json").read_text())
    assert m["n_tokens"] == 0 and m["finish_reason"] == "eos"
    assert (rollout_dir(u, 3, "alt1") / "output.txt").read_text() == "".join(f"<{i}>" for i in ids[:3])
    plan = one_site_plan(make_cond(), 19, ids[19], {"alt1": 5})
    go_forks(tiny_model, io_of(fake_io), tmp_path, plan)
    m = json.loads((rollout_dir(u, 19, "alt1") / "meta.json").read_text())
    assert m["n_tokens"] == 0 and m["finish_reason"] == "max_tokens"


def test_parent_identity_checked(tiny_model, fake_io, tmp_path, parent):
    u, ids = parent
    plan = one_site_plan(make_cond(), 3, ids[3], {"alt1": 7})
    with pytest.raises(IdentityMismatch):
        go_forks(tiny_model, io_of(fake_io), tmp_path, plan, ps=make_ps(text="changed"))
    last = json.loads((tmp_path / "runlog.jsonl").read_text().splitlines()[-1])
    assert last["error"].startswith("IdentityMismatch") and last["kind"] == "forks"


def test_plan_of_other_condition_rejected(tiny_model, fake_io, tmp_path, parent):
    u, ids = parent
    plan = one_site_plan(make_cond(), 3, ids[3], {"alt1": 7})
    plan["condition_id"] = "other-12345678"
    with pytest.raises(IdentityMismatch):
        go_forks(tiny_model, io_of(fake_io), tmp_path, plan)


def test_done_rollout_with_other_forced_token_is_rejected(tiny_model, fake_io, tmp_path, parent):
    u, ids = parent
    go_forks(tiny_model, io_of(fake_io), tmp_path, one_site_plan(make_cond(), 3, ids[3], {"alt1": 7}))
    with pytest.raises(IdentityMismatch):
        go_forks(tiny_model, io_of(fake_io), tmp_path, one_site_plan(make_cond(), 3, ids[3], {"alt1": 8}))


def test_same_seed_same_rollout(tiny_model, fake_io, tmp_path):
    outs = []
    for root in (tmp_path / "a", tmp_path / "b"):
        cond, ps = make_cond(), make_ps()
        run_units([Unit(ps.prompts[0], 0)], cond, ps, tiny_model, None, root, None,
                  fingerprint="test", now=Clock(), io=io_of(fake_io), prompt_ids_fn=lambda p: [1, 2, 3])
        u = run_dir(root, cond.condition_id, "p0", 0)
        ids = read_tokens(u).to_pydict()["token_id"]
        go_forks(tiny_model, io_of(fake_io), root, one_site_plan(cond, 3, ids[3], {"same1": ids[3], "same2": ids[3]}))
        outs.append([read_tokens(rollout_dir(u, 3, v)).to_pydict()["token_id"] for v in ("same1", "same2")])
    assert outs[0] == outs[1]


def test_text_split_with_merging_detokenizer(tiny_model, fake_io, tmp_path):
    def merging(ids):  # odd ids carry no text of their own; their char goes to the next token
        out, buf = [], ""
        for i in ids:
            if i % 2:
                buf += "~"
                out.append("")
            else:
                out.append(buf + f"<{i}>")
                buf = ""
        if buf and out:
            out[-1] += buf
        return out

    io = (fake_io["eos_ids"], merging)
    cond, ps = make_cond(), make_ps()
    run_units([Unit(ps.prompts[0], 0)], cond, ps, tiny_model, None, tmp_path, None,
              fingerprint="test", now=Clock(), io=io, prompt_ids_fn=lambda p: [1, 2, 3])
    u = run_dir(tmp_path, cond.condition_id, "p0", 0)
    ids = read_tokens(u).to_pydict()["token_id"]
    go_forks(tiny_model, io, tmp_path, one_site_plan(cond, 3, ids[3], {"alt1": 7}))
    d = rollout_dir(u, 3, "alt1")
    off = json.loads((d / "meta.json").read_text())["fork"]["continuation_char_offset"]
    t = read_tokens(d).to_pydict()
    out = (d / "output.txt").read_text()
    assert out[off:] == "".join(t["text"])
    assert all(t["char_offset"][k] == len("".join(t["text"][:k])) for k in range(len(t["text"])))


def test_parent_tokens_must_match_plan(tiny_model, fake_io, tmp_path, parent):
    u, ids = parent
    plan = one_site_plan(make_cond(), 3, ids[3] + 1, {"same1": ids[3] + 1})  # plan built on other tokens
    with pytest.raises(IdentityMismatch):
        go_forks(tiny_model, io_of(fake_io), tmp_path, plan)
    plan = one_site_plan(make_cond(), 3, ids[3], {"alt1": 7})
    plan["apps"][0]["n_tokens"] = len(ids) + 5
    with pytest.raises(IdentityMismatch):
        go_forks(tiny_model, io_of(fake_io), tmp_path, plan)
    assert not (u / "forks").exists()


def test_more_same_variants_than_default(tiny_model, fake_io, tmp_path, parent):
    u, ids = parent
    v = {"alt1": 7, **{f"same{k}": ids[3] for k in range(1, 7)}}
    r = go_forks(tiny_model, io_of(fake_io), tmp_path, one_site_plan(make_cond(), 3, ids[3], v))
    assert r["done"] == 7 and all(is_done(rollout_dir(u, 3, f"same{k}")) for k in range(1, 7))
