import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from entropytrail.novelty_cross import analyse, build_key, collect, server_map

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("serve_apps", ROOT / "scripts" / "serve_apps.py")
serve_apps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve_apps)


def _cond(tmp, name, prompts=("p1", "p2"), n=4):
    d = tmp / name
    for p in prompts:
        for s in range(n):
            u = d / p / f"{s:03d}"
            u.mkdir(parents=True)
            (u / "app.html").write_text(f"<html><head></head><body>{name} {p} {s}</body></html>")
    return d


def test_key_is_blind_shuffled_and_seeded(tmp_path):
    conds = {1.0: _cond(tmp_path, "c10"), 0.7: _cond(tmp_path, "c07")}
    key = build_key(conds, ["p1", "p2"], seed=3)
    assert key == build_key(conds, ["p1", "p2"], seed=3) and key != build_key(conds, ["p1", "p2"], seed=4)
    items = key["prompts"]["p1"]
    assert [it["blind"] for it in items] == [f"{i:03d}" for i in range(8)]
    assert {(it["temperature"], it["sample"]) for it in items} == {(t, s) for t in (1.0, 0.7) for s in range(4)}
    temps = [it["temperature"] for it in items]
    assert temps != sorted(temps) and temps != sorted(temps, reverse=True)  # interleaved, not blocks


def test_server_map_serves_blind_ids(tmp_path):
    conds = {1.0: _cond(tmp_path, "c10"), 0.7: _cond(tmp_path, "c07")}
    key = build_key(conds, ["p1"], seed=0)
    smap = server_map(key)
    assert "c07" not in json.dumps(list(smap["p1"].keys()))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), serve_apps.make_handler(tmp_path, blind=smap))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        it = key["prompts"]["p1"][5]
        body = urllib.request.urlopen(f"http://127.0.0.1:{srv.server_address[1]}/p1/005/").read().decode()
        tag = "c10" if it["temperature"] == 1.0 else "c07"
        assert f"{tag} p1 {it['sample']}" in body and "rating shim" in body
    finally:
        srv.shutdown()


def test_collect_maps_back_and_refuses_incomplete(tmp_path):
    conds = {1.0: _cond(tmp_path, "c10"), 0.7: _cond(tmp_path, "c07")}
    key = build_key(conds, ["p1"], seed=0)
    sc = tmp_path / "scores"
    sc.mkdir()
    scores = {f"{i:03d}": {"novelty": 1 + i % 5, "idea": f"idea {i}"} for i in range(8)}
    (sc / "p1.json").write_text(json.dumps({"scores": scores}))
    df = collect(key, sc)
    assert len(df) == 8 and set(df.columns) >= {"prompt_id", "temperature", "condition_id", "sample", "novelty_cross"}
    r = df[df["blind"] == "003"].iloc[0]
    assert r["novelty_cross"] == 4 and r["idea"] == "idea 3"
    del scores["007"]
    (sc / "p1.json").write_text(json.dumps({"scores": scores}))
    with pytest.raises(ValueError):
        collect(key, sc)


def test_analyse_detects_lower_novelty_at_low_t():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(12):
        base = rng.normal(0, 0.5)
        for t, shift in ((1.0, 0.0), (0.7, -0.5), (0.5, -1.0)):
            for s in range(20):
                rows.append({"prompt_id": f"p{i:02d}", "domain": ["education", "game", "tool"][i % 3],
                             "openness": "loose" if i % 2 else "tight", "temperature": t, "sample": s,
                             "novelty_cross": float(np.clip(round(3 + base + shift + rng.normal(0, 0.7)), 1, 5))})
    out = analyse(pd.DataFrame(rows))
    assert set(out["effects"]) == {"0.7", "0.5"}
    assert out["effects"]["0.5"]["estimate"] < out["effects"]["0.7"]["estimate"] < 0
    assert out["effects"]["0.5"]["p_one_sided"] < 0.001 and "exploratory" in out["label"]
