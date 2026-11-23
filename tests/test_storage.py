from pathlib import Path
import json

import numpy as np
import pyarrow as pa
import pytest

from entropytrail.stats import TOPK, TokenStats
from entropytrail.storage import (
    Recording, TokenRecord, clean_partials, data_root, is_done, read_tokens, run_dir, write_run,
)


def _stats(i):
    return TokenStats(
        H=1.5 + i, S=2.5 + i, varentropy=0.25 + i, margin=0.5 + i,
        topk_ids=np.arange(TOPK, dtype=np.int32) + i,
        topk_logp2=-np.arange(TOPK, dtype=np.float32) - i,
    )


def _rec(text="<html>x</html>"):
    texts = ["<ht", "ml>x", "</html>"]
    toks, off = [], 0
    for i, t in enumerate(texts):
        toks.append(TokenRecord(pos=i, token_id=100 + i, text=t, char_offset=off, stats=_stats(i)))
        off += len(t)
    return Recording(prompt_ids=[1, 2], tokens=toks, text=text, finish_reason="eos", seconds=2.0)


def test_round_trip(tmp_path):
    d = run_dir(tmp_path, "c", "p", 3)
    assert d == tmp_path / "c" / "p" / "003"
    assert not is_done(d)
    write_run(d, _rec(), {"seed": 7})
    assert is_done(d)
    assert not list(tmp_path.rglob("*.partial"))
    t = read_tokens(d)
    s = t.schema
    assert s.field("pos").type == pa.int32()
    assert s.field("token_id").type == pa.int32()
    assert s.field("text").type == pa.string()
    assert s.field("char_offset").type == pa.int32()
    for c in ("H", "S", "varentropy", "margin"):
        assert s.field(c).type == pa.float32()
    assert s.field("topk_ids").type == pa.list_(pa.int32())
    assert s.field("topk_logp2").type == pa.list_(pa.float32())
    assert t.num_rows == 3
    assert t["token_id"].to_pylist() == [100, 101, 102]
    assert t["H"].to_pylist() == [1.5, 2.5, 3.5]
    ids = t["topk_ids"].to_pylist()
    assert len(ids[1]) == TOPK and ids[1][0] == 1
    assert t["topk_logp2"].to_pylist()[2][1] == -3.0
    texts = t["text"].to_pylist()
    for i, off in enumerate(t["char_offset"].to_pylist()):
        assert off == len("".join(texts[:i]))
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 7 and meta["n_tokens"] == 3 and meta["status"] == "done"
    assert meta["finish_reason"] == "eos" and meta["seconds"] == 2.0
    assert meta["tokens_per_s"] == 1.5 and meta["html_extracted"] is True
    assert (d / "output.txt").read_text(encoding="utf-8") == "<html>x</html>"
    assert "<html>" in (d / "app.html").read_text(encoding="utf-8")


def test_no_html(tmp_path):
    d = run_dir(tmp_path, "c", "p", 0)
    write_run(d, _rec("nothing here ä"), {})
    assert not (d / "app.html").exists()
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["html_extracted"] is False
    assert (d / "output.txt").read_text(encoding="utf-8") == "nothing here ä"


def test_existing_raises_and_stale_partial_removed(tmp_path):
    d = run_dir(tmp_path, "c", "p", 1)
    stale = d.parent / (d.name + ".partial")
    stale.mkdir(parents=True)
    (stale / "junk").write_text("x")
    write_run(d, _rec(), {})
    assert not stale.exists()
    with pytest.raises(FileExistsError):
        write_run(d, _rec(), {})


def test_zero_seconds(tmp_path):
    r = _rec()
    r.seconds = 0.0
    d = run_dir(tmp_path, "c", "p", 0)
    write_run(d, r, {})
    assert json.loads((d / "meta.json").read_text())["tokens_per_s"] == 0.0


def test_clean_partials(tmp_path):
    p = tmp_path / "c" / "p" / "x.partial"
    p.mkdir(parents=True)
    (p / "f").write_text("1")
    assert clean_partials(tmp_path) == 1
    assert not p.exists()
    assert clean_partials(tmp_path) == 0


def test_data_root(monkeypatch, tmp_path):
    monkeypatch.delenv("ENTROPYTRAIL_DATA", raising=False)
    assert str(data_root()) == "/Volumes/Data/entropytrail"
    target = tmp_path / "new"
    monkeypatch.setenv("ENTROPYTRAIL_DATA", str(target))
    assert data_root() == target
    assert not target.exists()


def test_results_dir_per_condition(tmp_path):
    from entropytrail.storage import results_dir
    base = tmp_path / "results"
    assert results_dir(Path("/x/qwen35-9b-bf16-mlx-1195d88b"), base) == base
    assert results_dir(Path("/x/qwen35-9b-bf16-mlx-t0p7-abcd1234"), base) == base / "qwen35-9b-bf16-mlx-t0p7-abcd1234"
