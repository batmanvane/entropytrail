import json

import numpy as np
from PIL import Image

from entropytrail.evaluate import eval_units, find_units, load_outcome


def make_unit(root, prompt, sample, html, done=True, sub=None):
    u = root / "cond" / prompt / f"{sample:03d}"
    if sub:
        u = u / sub
    u.mkdir(parents=True)
    (u / "output.txt").write_text(html or "no html here")
    if html is not None:
        (u / "app.html").write_text(html)
    (u / "meta.json").write_text(json.dumps({"status": "done" if done else "running"}))
    return u


def fake_render(html, out):
    out.mkdir(parents=True, exist_ok=True)
    for f in ("load.png", "interact.png"):
        Image.new("RGB", (8, 8), "white").save(out / f)
    r = {"renders": html is not None, "render_version": 1}
    (out / "render.json").write_text(json.dumps(r))
    return r


def raise_on(bad):
    def fn(html, out):
        if out.parent == bad:
            raise RuntimeError("boom")
        return fake_render(html, out)
    return fn


class FakeEmb:
    model_id = "fake/model"

    def __init__(self):
        self.last_truncated = []

    def embed(self, items):
        self.last_truncated = [False] * len(items)
        return np.tile(np.array([[1.0, 0.0]], np.float32), (len(items), 1))


def run(units, render_fn=fake_render):
    return eval_units(units, render_fn=render_fn, image_embedder=FakeEmb(), code_embedder=FakeEmb())


def test_eval_writes_complete_folder_and_is_resumable(tmp_path):
    u = make_unit(tmp_path, "p1", 0, html="<html></html>")
    r1 = run([u])
    assert r1 == {"done": 1, "skipped": 0, "failed": 0}
    ev = u / "eval"
    assert {p.name for p in ev.iterdir()} == {
        "load.png", "interact.png", "render.json", "fingerprint.json", "embeddings.npz", "eval.json"}
    assert set(np.load(ev / "embeddings.npz").files) == {"img_load", "img_interact", "screen", "code"}
    meta = json.loads((ev / "eval.json").read_text())
    assert meta["status"] == "done" and meta["image_model"] == "fake/model" and meta["render_version"] == 1
    assert run([u]) == {"done": 0, "skipped": 1, "failed": 0}


def test_eval_redone_when_model_changes(tmp_path):
    u = make_unit(tmp_path, "p1", 0, html="<html></html>")
    run([u])
    meta = json.loads((u / "eval" / "eval.json").read_text())
    meta["image_model"] = "old/model"
    (u / "eval" / "eval.json").write_text(json.dumps(meta))
    assert run([u])["done"] == 1


def test_unit_without_app_still_gets_outcome(tmp_path):
    u = make_unit(tmp_path, "p1", 1, html=None)
    run([u])
    o = load_outcome(u)
    assert o.renders is False and np.linalg.norm(o.screen) == np.float32(1.0)


def test_failing_unit_does_not_abort_batch(tmp_path):
    a, b = make_unit(tmp_path, "p1", 0, html="x"), make_unit(tmp_path, "p1", 1, html="y")
    r = run([a, b], render_fn=raise_on(a))
    assert r == {"done": 1, "skipped": 0, "failed": 1}
    assert not (a / "eval").exists() and not (a / "eval.partial").exists()
    assert (b / "eval" / "eval.json").exists()


def test_find_units_includes_rollouts_only_with_flag(tmp_path):
    app = make_unit(tmp_path, "p1", 0, html="x")
    make_unit(tmp_path, "p1", 1, html="x", done=False)
    make_unit(tmp_path, "smoke-tool-01", 0, html="x")
    roll = make_unit(tmp_path, "p1", 0, html="x", sub="forks/00012/alt1")
    make_unit(tmp_path, "p1", 0, html="x", sub="forks/00012/alt1.stale-20260101-000000")
    make_unit(tmp_path, "p1", 0, html="x", sub="forks.smoke-20261122/00012/alt1")
    (tmp_path / "cond" / "p1" / "002.stale-20260101-000000").mkdir()
    (tmp_path / "cond" / "p1" / "003.partial").mkdir()
    cond = tmp_path / "cond"
    assert find_units(cond, prompt_ids=None, forks=False) == [app]
    assert find_units(cond, prompt_ids=None, forks=True) == [app, roll]
    assert find_units(cond, prompt_ids=["p2"], forks=True) == []
