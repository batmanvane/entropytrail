"""Per-app landscape measures (frozen analysis plan, METHODS.md; D-029; R3 in plan 2).

Only the app part counts (tokens whose first character lies in the extracted HTML).
fork = code token with H >= 2 bits; comment fork = comment token with H >= 2 bits.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from entropytrail.sections import CODE_SECTIONS, token_sections
from entropytrail.storage import read_tokens

FORK_H = 2.0
SURPRISE_S = 4.0
LOW_H = 0.1


def app_tokens(unit: Path) -> pd.DataFrame:
    t = read_tokens(unit).select(["pos", "char_offset", "H", "S"]).to_pandas()
    out = (Path(unit) / "output.txt").read_text(encoding="utf-8")
    t["section"] = token_sections(out, t["char_offset"].tolist(), detail=True)
    df = t[t["section"] != "outside"].drop(columns="char_offset").reset_index(drop=True)
    df["is_code"] = df["section"].isin(CODE_SECTIONS)
    n = len(df)
    df["rel_pos"] = np.arange(n) / n if n else []
    return df


def _late_js_surprise(df: pd.DataFrame) -> float:
    js = df[df["section"] == "js"]  # JS code tokens; JS comments are dropped before cutting (R3)
    n = len(js)
    if n < 3:
        return math.nan
    late = js.iloc[math.ceil(2 * n / 3):]
    return float((late["S"] >= SURPRISE_S).mean())


def app_measures(df: pd.DataFrame, n_chars: int, n_lines: int) -> dict:
    n = len(df)
    code = df[df["is_code"]]
    comment = df[df["section"].str.endswith("comment")]
    forks = code[code["H"] >= FORK_H]
    top = forks.nlargest(10, "H")
    sum_H, sum_S = float(df["H"].sum()), float(df["S"].sum())
    return {
        "n_app_tokens": n,
        "mean_H": sum_H / n if n else math.nan,
        "mean_S": sum_S / n if n else math.nan,
        "share_H_lt_0_1": float((df["H"] < LOW_H).mean()) if n else math.nan,
        "n_forks": len(forks),
        "fork_share": len(forks) / len(code) if len(code) else math.nan,
        "comment_fork_share": float((comment["H"] >= FORK_H).mean()) if len(comment) else math.nan,
        "fork_surprise": float(forks["S"].mean()) if len(forks) else math.nan,
        "late_js_surprise": _late_js_surprise(df),
        "path_action": sum_S,
        "path_action_per_token": sum_S / n if n else math.nan,
        "H_per_char": sum_H / n_chars if n_chars else math.nan,
        "H_per_line": sum_H / n_lines if n_lines else math.nan,
        "median_relpos_top10_forks": float(top["rel_pos"].median()) if len(top) else math.nan,
    }
