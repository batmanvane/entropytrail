import pytest

pytest.importorskip("PIL")
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("collect_outcomes", ROOT / "scripts" / "collect_outcomes.py")
co = importlib.util.module_from_spec(spec)
spec.loader.exec_module(co)


def unit(cond, prompt, sample, screen, renders=True, html="<html><p>a</p>\n</html>", code=(1, 0)):
    u = cond / prompt / f"{sample:03d}"
    (u / "eval").mkdir(parents=True)
    (u / "meta.json").write_text(json.dumps({
        "status": "done", "n_tokens": 10, "finish_reason": "eos", "html_extracted": html is not None,
        "prompt": {"id": prompt, "domain": "game", "openness": "tight"}}))
    if html is not None:
        (u / "app.html").write_text(html)
    s = np.asarray(screen, np.float32)
    s /= np.linalg.norm(s)
    np.savez(u / "eval" / "embeddings.npz", img_load=s, img_interact=s, screen=s,
             code=np.asarray(code, np.float32))
    (u / "eval" / "render.json").write_text(json.dumps({
        "renders": renders, "timeout": False, "n_page_errors": 0 if renders else 1,
        "n_console_errors": 0, "n_page_errors_after_interaction": 0}))
    (u / "eval" / "fingerprint.json").write_text(json.dumps({"tags": {"p": 1}, "js_functions": [], "css_colors": []}))
    (u / "eval" / "eval.json").write_text(json.dumps({"status": "done"}))


def test_collect_rows(tmp_path):
    cond = tmp_path / "c"
    unit(cond, "g1", 0, [1, 0])
    unit(cond, "g1", 1, [1, 0], renders=False)
    unit(cond, "g2", 0, [1, 0])
    unit(cond, "g2", 1, [0, 1], html=None)
    apps, ens = co.collect(cond)
    assert len(apps) == 4 and len(ens) == 2
    g1 = next(e for e in ens if e["prompt_id"] == "g1")
    g2 = next(e for e in ens if e["prompt_id"] == "g2")
    assert g1["ensemble_spread"] == 0.0 and g1["ensemble_spread_render_rule"] == 1.0
    assert g1["n_renders"] == 1 and g1["code_spread"] == 0.0 and g1["fingerprint_spread"] == 0.0
    assert g2["ensemble_spread"] == pytest.approx(1.0)
    a = next(r for r in apps if r["prompt_id"] == "g2" and r["sample"] == "001")
    assert a["n_chars"] == 0 and a["n_lines"] == 0 and a["ensemble_rarity"] == pytest.approx(1.0)
    b = next(r for r in apps if r["prompt_id"] == "g1" and r["sample"] == "000")
    assert b["n_lines"] == 2 and b["domain"] == "game" and b["renders"] is True


def test_missing_eval_is_an_error(tmp_path):
    cond = tmp_path / "c"
    unit(cond, "g1", 0, [1, 0])
    (cond / "g1" / "001").mkdir()
    (cond / "g1" / "001" / "meta.json").write_text(json.dumps({"status": "done"}))
    with pytest.raises(FileNotFoundError):
        co.collect(cond)
