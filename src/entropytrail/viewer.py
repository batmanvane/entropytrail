"""Entropy-coloured source viewer: one standalone HTML page per app (WP6, spec §9)."""
from __future__ import annotations

import html as _html
from typing import Callable

# warm-gray ramp, paper (certain) -> dark (open), as in the README figures
RAMP = ["#f7f6f2", "#e3e0d8", "#c4c1b8", "#9d9a92", "#77746d", "#4a4844"]
H_MAX = 3.0

CSS = """
:root{--paper:#f2f1ec;--ink:#1a1a1a;--accent:#1f4d3a}
body{background:var(--paper);color:var(--ink);font:13px/1.5 "IBM Plex Mono",Menlo,monospace;margin:16px}
pre{white-space:pre-wrap;word-break:break-word}
.t{border-radius:2px}.fork{outline:2px solid var(--accent)}
.light{color:#f7f6f2}
"""


def _colour(h: float) -> tuple[str, bool]:
    i = min(len(RAMP) - 1, int(max(0.0, h) / H_MAX * (len(RAMP) - 1) + 0.5))
    return RAMP[i], i >= 4


def render_viewer(df, forks: list[int], decode: Callable[[int], str] | None = None, title: str = "") -> str:
    fork_set = set(forks)
    parts = []
    for i, row in enumerate(df.itertuples(index=False)):
        if not row.text:
            continue
        bg, light = _colour(float(row.H))
        alts = ""
        if decode is not None:
            alts = " | " + ", ".join(f"{decode(int(t))!r} {2 ** float(lp):.2f}"
                                     for t, lp in list(zip(row.topk_ids, row.topk_logp2))[:5])
        tip = _html.escape(f"H {row.H:.2f} bits, S {row.S:.2f} bits{alts}", quote=True)
        cls = "t fork" if i in fork_set else "t"
        cls += " light" if light else ""
        parts.append(f'<span class="{cls}" style="background:{bg}" title="{tip}">{_html.escape(row.text)}</span>')
    return (f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width'>"
            f"<title>{_html.escape(title)}</title><style>{CSS}</style></head><body><pre>"
            + "".join(parts) + "</pre></body></html>")
