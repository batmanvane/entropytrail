"""Stopping and entropy over position from the rescoring pass (D-042, exploratory).

Per app (grid unit with ``rescore.parquet``):

- ``premature_stop``: probability that sampling at the condition temperature would have drawn
  an end token while the HTML document was still open (before the final ``</html>`` was
  complete): 1 - prod(1 - P_T(stop)) over those positions. ``premature_stop_T1`` the same
  under the untempered model.
- ``p_stop_after_close``: P_T(stop) at the first position after ``</html>``; ``n_tail``:
  tokens from there to the end (closing fence, end token).
- ``p_stop_final``: P_T(stop) at the position where the end token was drawn.

Curves (boilerplate control for the lock-in question): mean H and H_T of code tokens by
relative position in the app (10 bins), separately per section (html, css, js, text), plus
within-JS position; and per app the slope of H over relative position within JS code, with
and without the last 10 % of JS tokens.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from entropytrail.sections import CODE_SECTIONS, token_sections
from entropytrail.storage import read_tokens

LABEL = ("exploratory (D-042) - teacher-forced stop probabilities and entropy over position; "
         "defined after the frozen analyses; not part of any Holm family")
BINS = 10


def _close_end(output: str) -> int | None:
    i = output.lower().rfind("</html>")
    return None if i < 0 else i + len("</html>")


def _no_stop(p: np.ndarray) -> float:
    """1 - prod(1 - p), stable for many small p."""
    return float(-np.expm1(np.sum(np.log1p(-np.clip(p, 0.0, 1.0 - 1e-12))))) if len(p) else 0.0


def _slope(y: np.ndarray) -> float:
    if len(y) < 20:
        return math.nan
    x = np.arange(len(y)) / len(y)
    return float(np.polyfit(x, y, 1)[0])


def unit_stop(unit: Path) -> tuple[dict, pd.DataFrame]:
    unit = Path(unit)
    t = read_tokens(unit).select(["pos", "char_offset", "H"]).to_pandas()
    r = pq.read_table(unit / "rescore.parquet").to_pandas()
    if len(r) != len(t) or not (r["pos"].to_numpy() == t["pos"].to_numpy()).all():
        raise ValueError(f"{unit}: rescore does not match tokens")
    df = t.join(r.drop(columns="pos"))
    df["p_T"] = np.exp2(df["logp2_eos_T"].astype(np.float64))
    df["p_1"] = np.exp2(df["logp2_eos"].astype(np.float64))
    out = (unit / "output.txt").read_text(encoding="utf-8")
    df["section"] = token_sections(out, df["char_offset"].tolist(), detail=True)
    close = _close_end(out)
    n = len(df)
    row: dict = {"n_tokens": n}
    if close is None:
        row.update(premature_stop=math.nan, premature_stop_T1=math.nan, p_stop_after_close=math.nan,
                   n_tail=math.nan)
    else:
        open_ = df["char_offset"].to_numpy() < close
        row.update(premature_stop=_no_stop(df.loc[open_, "p_T"].to_numpy()),
                   premature_stop_T1=_no_stop(df.loc[open_, "p_1"].to_numpy()),
                   n_tail=int((~open_).sum()))
        after = np.flatnonzero(~open_)
        row["p_stop_after_close"] = float(df["p_T"].iloc[after[0]]) if len(after) else math.nan
    row["p_stop_final"] = float(df["p_T"].iloc[-1]) if n else math.nan

    app = df[df["section"] != "outside"].reset_index(drop=True)
    app["rel_pos"] = np.arange(len(app)) / max(len(app), 1)
    code = app[app["section"].isin(CODE_SECTIONS)].copy()
    code["bin"] = np.minimum((code["rel_pos"] * BINS).astype(int), BINS - 1)
    js = code[code["section"] == "js"]
    row["js_slope_H"] = _slope(js["H"].to_numpy())
    row["js_slope_H_trim10"] = _slope(js["H"].to_numpy()[: int(len(js) * 0.9)])
    curves = code.groupby(["section", "bin"]).agg(H=("H", "mean"), H_T=("H_T", "mean"),
                                                 n=("H", "size")).reset_index()
    if len(js):
        jb = js.assign(bin=np.minimum((np.arange(len(js)) / len(js) * BINS).astype(int), BINS - 1))
        jc = jb.groupby("bin").agg(H=("H", "mean"), H_T=("H_T", "mean"), n=("H", "size")).reset_index()
        curves = pd.concat([curves, jc.assign(section="js-within")], ignore_index=True)
    allc = code.groupby("bin").agg(H=("H", "mean"), H_T=("H_T", "mean"), n=("H", "size")).reset_index()
    curves = pd.concat([curves, allc.assign(section="all-code")], ignore_index=True)
    return row, curves


def summarise(apps: pd.DataFrame, curves: pd.DataFrame) -> dict:
    """apps: one row per app with temperature, n_tokens, finish_reason and the unit_stop fields;
    curves: unit curves with temperature (per-app means are averaged with equal app weight)."""
    per_t = {}
    for t, g in apps.groupby("temperature"):
        q = g["n_tokens"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        per_t[str(t)] = {
            "n_apps": len(g),
            "finish_reason": g["finish_reason"].value_counts().to_dict(),
            "length": {"mean": float(g["n_tokens"].mean()), "sd": float(g["n_tokens"].std()),
                       "quantiles": {str(k): float(v) for k, v in q.items()},
                       "max": int(g["n_tokens"].max()),
                       "most_common_length_count": int(g["n_tokens"].value_counts().iloc[0])},
            "premature_stop": {"mean": float(g["premature_stop"].mean()),
                               "median": float(g["premature_stop"].median()),
                               "share_gt_0_1": float((g["premature_stop"] > 0.1).mean())},
            "premature_stop_T1_median": float(g["premature_stop_T1"].median()),
            "p_stop_after_close_median": float(g["p_stop_after_close"].median()),
            "n_tail_median": float(g["n_tail"].median()),
            "p_stop_final_median": float(g["p_stop_final"].median()),
            "js_slope_H": {"median": float(g["js_slope_H"].median()),
                           "share_negative": float((g["js_slope_H"].dropna() < 0).mean())},
            "js_slope_H_trim10": {"median": float(g["js_slope_H_trim10"].median()),
                                  "share_negative": float((g["js_slope_H_trim10"].dropna() < 0).mean())},
        }
    cur = (curves.groupby(["temperature", "section", "bin"])[["H", "H_T"]].mean().reset_index())
    return {"label": LABEL, "per_temperature": per_t,
            "curves": cur.to_dict(orient="records")}
