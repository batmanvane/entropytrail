import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("drs", ROOT / "scripts" / "draw_rating_sample.py")
drs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drs)


def test_draw_is_stratified_and_seeded():
    o = pd.DataFrame([{"prompt_id": p, "sample": s} for p in ("a", "b", "c") for s in range(20)])
    d = drs.draw(o)
    assert len(d) == 15 and (d.groupby("prompt_id").size() == 5).all()
    assert d.equals(drs.draw(o)) and not d.equals(drs.draw(o, seed=1))
    assert not d.duplicated().any()
