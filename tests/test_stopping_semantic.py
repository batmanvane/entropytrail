import json
import math

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from entropytrail.analysis.semantic import analyse, entropy_bits, load_partition, semantic_table
from entropytrail.analysis.stopping import summarise, unit_stop


def _unit(tmp_path, pieces, p_stop):
    """pieces: token texts; p_stop: P_T(stop) per position."""
    u = tmp_path / "c" / "p1" / "000"
    u.mkdir(parents=True)
    out = "".join(pieces)
    offs = np.cumsum([0] + [len(p) for p in pieces[:-1]])
    n = len(pieces)
    pq.write_table(pa.table({"pos": pa.array(np.arange(n, dtype=np.int32)), "char_offset": pa.array(offs.astype(np.int32)),
                             "H": pa.array(np.linspace(2, 0, n).astype(np.float32))}), u / "tokens.parquet")
    lp = np.log2(np.asarray(p_stop, dtype=np.float64)).astype(np.float32)
    pq.write_table(pa.table({"pos": pa.array(np.arange(n, dtype=np.int32)), "H_T": pa.array(np.ones(n, np.float32)),
                             "logp2_eos": lp, "logp2_eos_T": lp}), u / "rescore.parquet")
    (u / "output.txt").write_text(out)
    return u


def test_unit_stop_premature_and_tail(tmp_path):
    js = [f"let a{i}=1;" for i in range(30)]
    pieces = ["```html\n", "<html>", "<script>", *js, "</script>", "</html>", "\n```", ""]
    p = [0.01] * (len(pieces) - 3) + [0.5, 0.9, 0.99]
    u = _unit(tmp_path, pieces, p)
    row, curves = unit_stop(u)
    n_open = len(pieces) - 2  # positions before "</html>" is complete: up to and including the one writing it
    assert row["n_tail"] == 2 and math.isclose(row["p_stop_after_close"], 0.9, rel_tol=1e-5)
    assert math.isclose(row["premature_stop"], 1 - 0.99 ** (n_open - 1) * 0.5, rel_tol=1e-4)
    assert math.isclose(row["p_stop_final"], 0.99, rel_tol=1e-5)
    assert row["js_slope_H"] < 0
    assert {"js", "js-within", "all-code"} <= set(curves["section"])
    apps = pd.DataFrame([dict(row, temperature=1.0, finish_reason="eos")])
    s = summarise(apps, curves.assign(temperature=1.0))
    assert s["per_temperature"]["1.0"]["n_apps"] == 1 and "exploratory" in s["label"]


def test_entropy_bits():
    assert entropy_bits([5, 5]) [0] == pytest.approx(1.0)
    assert entropy_bits([20])[0] == 0.0
    h, mm = entropy_bits([10, 5, 5])
    assert h == pytest.approx(1.5) and mm == pytest.approx(1.5 + 2 / (40 * math.log(2)))


def _key(n_prompts=6):
    key = {"prompts": {}}
    for k in range(n_prompts):
        items = []
        for i, t in enumerate([1.0] * 4 + [0.5] * 4):
            items.append({"blind": f"{i:03d}", "temperature": t})
        key["prompts"][f"p{k}"] = items
    return key


def test_partition_and_table(tmp_path):
    key = _key()
    for pid in key["prompts"]:
        # T=1.0 apps (000-003) in 4 classes; T=0.5 apps (004-007) all in one class
        cl = [{"label": "a", "ids": ["000", "004", "005", "006", "007"]}, {"label": "b", "ids": ["001"]},
              {"label": "c", "ids": ["002"]}, {"label": "d", "ids": ["003"]}]
        (tmp_path / f"{pid}.json").write_text(json.dumps({"clusters": cl}))
    tab = semantic_table(key, tmp_path)
    r = tab[(tab.prompt_id == "p0") & (tab.temperature == 0.5)].iloc[0]
    assert r["H_sem"] == 0 and r["n_clusters"] == 1 and r["n_clusters_pool"] == 4
    assert tab[(tab.prompt_id == "p0") & (tab.temperature == 1.0)].iloc[0]["H_sem"] == pytest.approx(2.0)
    res = analyse(tab)
    assert res["effects"]["0.5"]["mean_diff_bits"] == pytest.approx(-2.0)
    assert res["effects"]["0.5"]["wilcoxon_p_one_sided_lower"] < 0.05
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"clusters": [{"ids": ["000", "000"]}]}))
    with pytest.raises(ValueError):
        load_partition(bad, {"000"})
    bad.write_text(json.dumps({"clusters": [{"ids": ["000"]}]}))
    with pytest.raises(ValueError):
        load_partition(bad, {"000", "001"})
