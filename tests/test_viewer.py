import pytest

pytest.importorskip("pandas")
import pandas as pd

from entropytrail.viewer import render_viewer


def frame():
    return pd.DataFrame({
        "text": ["<div>", " a&b", "", "x"], "H": [0.0, 2.5, 0.0, 4.0], "S": [0.0, 1.0, 0.0, 3.0],
        "topk_ids": [[1, 2], [3, 4], [5], [6, 7]], "topk_logp2": [[-0.1, -4], [-1, -1.2], [0], [-1, -2]],
    })


def test_one_span_per_nonempty_token_and_escaping():
    html = render_viewer(frame(), forks=[1], decode=lambda i: f"t{i}", title="p1/000")
    assert html.count('<span class="t') == 3
    assert "&lt;div&gt;" in html and " a&amp;b" in html and "<div>" not in html.split("<body>")[1]
    assert html.count("t fork") == 1
    assert "t3" in html and "H 2.50" in html


def test_colour_ramp_capped():
    html = render_viewer(frame(), forks=[], title="x")
    assert html.startswith("<!doctype html>") and "<title>x</title>" in html
