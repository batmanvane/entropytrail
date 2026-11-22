import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from entropytrail.forks import (
    VARIANTS, alternatives, build_plan, choose_apps, read_plan, select_sites, write_plan,
)


def test_top_sites_respect_spacing_and_code_only():
    H = np.zeros(200)
    H[[50, 55, 120]] = [5.0, 4.9, 3.0]
    H[10] = 9.0
    sec = ["js"] * 200
    sec[10] = "comment"
    assert select_sites(H, sec, n_top=2, n_random=0, min_spacing=20, seed=0) == [(50, "top"), (120, "top")]


def test_random_sites_seeded_and_distinct():
    H = np.random.default_rng(1).random(500)
    sec = ["css"] * 500
    a = select_sites(H, sec, seed=7)
    b = select_sites(H, sec, seed=7)
    assert a == b and sum(k == "random" for _, k in a) == 3 and len({p for p, _ in a}) == 13
    pos = sorted(p for p, _ in a)
    assert [p for p, _ in a] == pos and min(np.diff(pos)) >= 20
    assert select_sites(H, sec, seed=8) != a


def test_last_token_and_outside_excluded():
    H = np.zeros(30)
    H[29], H[0] = 9, 8
    sec = ["outside"] + ["js"] * 29
    sites = select_sites(H, sec, n_top=1, n_random=0, seed=0)
    assert all(p not in (0, 29) for p, _ in sites)


def test_few_candidates_give_fewer_sites():
    assert len(select_sites(np.ones(10), ["js"] * 10, seed=0)) == 1


def test_alternatives_skip_actual():
    assert alternatives([5, 7, 9, 11], actual_id=7) == (5, 9, 11)
    assert alternatives([7, 5], actual_id=7) == (5,)


def _unit(cond, prompt, sample, n=120, html=True):
    u = cond / prompt / f"{sample:03d}"
    u.mkdir(parents=True)
    texts = ["Here:\n"] + ["<html>"] + ["<script>"] + [f"a{i};" for i in range(n - 4)] + ["</script></html>"]
    out = "".join(texts)
    offs = np.cumsum([0] + [len(t) for t in texts[:-1]]).tolist()
    rng = np.random.default_rng(sample)
    H = rng.random(n).astype(np.float32) * 4
    tbl = pa.table({
        "pos": pa.array(range(n), pa.int32()), "token_id": pa.array([100 + i for i in range(n)], pa.int32()),
        "text": texts, "char_offset": pa.array(offs, pa.int32()), "H": pa.array(H), "S": pa.array(H / 2),
        "topk_ids": [[100 + i, 1, 2, 3] for i in range(n)],
    })
    pq.write_table(tbl, u / "tokens.parquet")
    (u / "output.txt").write_text(out)
    (u / "meta.json").write_text(json.dumps({"status": "done", "html_extracted": html, "n_tokens": n}))
    return u


def test_choose_apps_seeded_and_only_extracted(tmp_path):
    cond = tmp_path / "c"
    for s in range(6):
        _unit(cond, "p1", s, html=s != 2)
    a = choose_apps(cond, ["p1"], 3, seed=1)
    assert a == choose_apps(cond, ["p1"], 3, seed=1) and len(a) == 3
    assert all(p == "p1" and s != 2 for p, s in a) and a == sorted(a)


def test_plan_roundtrip(tmp_path):
    cond = tmp_path / "qwen-x"
    cond.mkdir()
    (cond / "condition.json").write_text(json.dumps({"condition_id": "qwen-x"}))
    _unit(cond, "p1", 0)
    plan = build_plan(cond, [("p1", 0)], seed=5)
    write_plan(tmp_path / "plan.json", plan)
    back = read_plan(tmp_path / "plan.json")
    assert back == plan and back["condition_id"] == "qwen-x" and back["seed"] == 5
    app = back["apps"][0]
    assert app["prompt_id"] == "p1" and app["sample"] == 0 and 1 <= len(app["sites"]) <= 13
    for s in app["sites"]:
        assert list(s["variants"]) == list(VARIANTS)
        assert s["variants"]["same1"] == s["variants"]["same2"] == s["actual_id"] == 100 + s["pos"]
        assert s["actual_id"] not in (s["variants"]["alt1"], s["variants"]["alt2"], s["variants"]["alt3"])
        assert s["section"] == "js" and 0 <= s["rel_pos"] < 1
