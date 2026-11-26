import json
import math

import pytest

pytest.importorskip("pandas")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from entropytrail.analysis.measures import app_measures, app_tokens

SEC = ["html", "css", "css-comment", "css-comment", "js", "js", "js-comment", "js", "js", "js", "text", "html"]
H = [0.0, 2.5, 3.0, 0.05, 0.0, 2.0, 2.2, 0.0, 0.0, 3.0, 0.01, 0.0]
S = [0.0, 1.0, 2.0, 0.0, 5.0, 4.5, 6.0, 0.0, 4.0, 0.5, 0.0, 0.0]


def frame():
    from entropytrail.sections import CODE_SECTIONS
    n = len(SEC)
    return pd.DataFrame({"pos": range(n), "H": H, "S": S, "section": SEC,
                         "is_code": [s in CODE_SECTIONS for s in SEC], "rel_pos": [i / n for i in range(n)]})


def test_measures():
    m = app_measures(frame(), n_chars=100, n_lines=10)
    assert m["n_app_tokens"] == 12 and m["n_forks"] == 3
    assert m["fork_share"] == pytest.approx(3 / 9)
    assert m["comment_fork_share"] == pytest.approx(2 / 3)
    assert m["fork_surprise"] == pytest.approx(2.0)
    assert m["share_H_lt_0_1"] == pytest.approx(7 / 12)
    assert m["path_action"] == pytest.approx(23.0) and m["path_action_per_token"] == pytest.approx(23 / 12)
    assert m["mean_H"] == pytest.approx(12.76 / 12) and m["mean_S"] == pytest.approx(23 / 12)
    assert m["H_per_char"] == pytest.approx(0.1276) and m["H_per_line"] == pytest.approx(1.276)
    assert m["median_relpos_top10_forks"] == pytest.approx(5 / 12)


def test_late_js_surprise_drops_comments_first():
    # JS code tokens 4,5,7,8,9 -> last third (index >= ceil(10/3) = 4) is token 9 with S 0.5 -> 0.0
    # (the reading that keeps JS comments before cutting would give 0.5)
    assert app_measures(frame(), 100, 10)["late_js_surprise"] == 0.0
    f = frame()
    f.loc[9, "S"] = 4.0
    assert app_measures(f, 100, 10)["late_js_surprise"] == 1.0


def test_no_forks_and_no_js_give_nan():
    f = frame()
    f["H"] = 0.0
    f["section"] = ["html"] * 12
    f["is_code"] = True
    m = app_measures(f, 100, 10)
    assert m["n_forks"] == 0 and math.isnan(m["fork_surprise"]) and math.isnan(m["late_js_surprise"])
    assert math.isnan(m["median_relpos_top10_forks"])


def test_app_tokens_only_app_part(tmp_path):
    texts = ["Sure:\n", "<html>", "<script>", "// c\n", "x=1;", "</script></html>", "\nbye"]
    offs = np.cumsum([0] + [len(t) for t in texts[:-1]]).tolist()
    n = len(texts)
    pq.write_table(pa.table({"pos": list(range(n)), "text": texts, "char_offset": offs,
                             "H": [float(i) for i in range(n)], "S": [0.0] * n}), tmp_path / "tokens.parquet")
    (tmp_path / "output.txt").write_text("".join(texts))
    df = app_tokens(tmp_path)
    assert df["pos"].tolist() == [1, 2, 3, 4, 5]
    assert df["section"].tolist() == ["html", "html", "js-comment", "js", "html"]
    assert df["is_code"].tolist() == [True, True, False, True, True]
    assert df["rel_pos"].tolist() == pytest.approx([0, 0.2, 0.4, 0.6, 0.8])


def test_collect_measures_rows(tmp_path):
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("collect_measures", root / "scripts" / "collect_measures.py")
    cm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cm)
    cond = tmp_path / "c"
    for s, html in ((0, True), (1, False)):
        u = cond / "p1" / f"{s:03d}"
        u.mkdir(parents=True)
        texts = ["<html>", "<script>", "x=1;", "</script></html>"] if html else ["sorry"]
        offs = np.cumsum([0] + [len(t) for t in texts[:-1]]).tolist()
        n = len(texts)
        pq.write_table(pa.table({"pos": list(range(n)), "text": texts, "char_offset": offs,
                                 "H": [2.5] * n, "S": [1.0] * n}), u / "tokens.parquet")
        (u / "output.txt").write_text("".join(texts))
        if html:
            (u / "app.html").write_text("".join(texts))
        (u / "meta.json").write_text(json.dumps({"status": "done", "finish_reason": "eos", "html_extracted": html,
                                                 "prompt": {"id": "p1", "domain": "game", "openness": "tight"}}))
    rows = cm.collect(cond)
    assert [r["sample"] for r in rows] == ["000", "001"]
    assert rows[0]["n_app_tokens"] == 4 and rows[0]["fork_share"] == 1.0 and rows[0]["domain"] == "game"
    assert rows[1]["n_app_tokens"] is None and rows[1]["html_extracted"] is False
