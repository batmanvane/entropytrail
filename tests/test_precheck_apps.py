import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("precheck_apps", ROOT / "scripts" / "precheck_apps.py")
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)


def _unit(tmp_path, html, name="000"):
    u = tmp_path / name
    u.mkdir()
    (u / "meta.json").write_text("{}")
    if html is not None:
        (u / "app.html").write_text(html)
    return u


def test_parsing_app_is_left_for_chrome(tmp_path):
    u = _unit(tmp_path, "<html><script>let a = 1; function f(){ return a }</script></html>")
    assert pc.precheck_unit(u) == "parsed"
    assert not (u / "rating-claude.json").exists()


def test_syntax_error_gets_auto_rating(tmp_path):
    u = _unit(tmp_path, "<html><script>const g = { gcd: (a, b) { return a } };</script></html>")
    assert pc.precheck_unit(u) == "syntax"
    r = json.loads((u / "rating-claude.json").read_text())
    assert r["rater"] == "auto-precheck" and r["functionality"] == 1 and r["ui"] is None and r["novelty"] is None


def test_missing_html_gets_auto_rating_and_existing_rating_is_kept(tmp_path):
    u = _unit(tmp_path, None)
    assert pc.precheck_unit(u) == "syntax"
    assert pc.precheck_unit(u) == "rated"


def test_unit_still_generating_is_pending(tmp_path):
    u = tmp_path / "001"
    u.mkdir()
    assert pc.precheck_unit(u) == "pending"
