"""Section labels per character and per token (spec §5.3, D-029 comment rule).

Labels: ``html`` (inside tags), ``text`` (between tags), ``css`` (inside <style>),
``js`` (inside <script>), ``comment`` (JS ``//``/``/* */``, CSS ``/* */``, HTML ``<!-- -->``).
With ``detail=True`` comments are ``js-comment``, ``css-comment`` or ``html-comment``.
In JS, comment markers inside '…', "…" and `…` strings are code; regex literals are not
detected (a ``//`` inside a regex literal would start a comment).
"""
from __future__ import annotations

import re

CODE_SECTIONS = frozenset({"html", "css", "js", "text"})

_RAW_OPEN = re.compile(r"<(script|style)\b[^>]*>", re.IGNORECASE)


def _js(src: str, a: int, b: int, lab: list[str], com: str) -> None:
    i = a
    while i < b:
        c = src[i]
        if c in "'\"`":
            j = i + 1
            while j < b and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            j = min(j + 1, b)
            lab[i:j] = ["js"] * (j - i)
            i = j
        elif src.startswith("//", i):
            j = src.find("\n", i, b)
            j = b if j < 0 else j
            lab[i:j] = [com] * (j - i)
            i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2, b)
            j = b if j < 0 else j + 2
            lab[i:j] = [com] * (j - i)
            i = j
        else:
            lab[i] = "js"
            i += 1


def _css(src: str, a: int, b: int, lab: list[str], com: str) -> None:
    i = a
    while i < b:
        if src.startswith("/*", i):
            j = src.find("*/", i + 2, b)
            j = b if j < 0 else j + 2
            lab[i:j] = [com] * (j - i)
            i = j
        else:
            lab[i] = "css"
            i += 1


def char_sections(html: str, *, detail: bool = False) -> list[str]:
    n = len(html)
    lab = ["text"] * n
    jc, cc, hc = ("js-comment", "css-comment", "html-comment") if detail else ("comment",) * 3
    i = 0
    while i < n:
        if html.startswith("<!--", i):
            j = html.find("-->", i + 4)
            j = n if j < 0 else j + 3
            lab[i:j] = [hc] * (j - i)
            i = j
            continue
        if html[i] == "<":
            m = _RAW_OPEN.match(html, i)
            j = html.find(">", i)
            j = n if j < 0 else j + 1
            lab[i:j] = ["html"] * (j - i)
            if m:
                kind = m.group(1).lower()
                close = re.compile(rf"</{kind}\s*>", re.IGNORECASE).search(html, m.end())
                b = close.start() if close else n
                (_js if kind == "script" else _css)(html, m.end(), b, lab, jc if kind == "script" else cc)
                if close:
                    lab[close.start():close.end()] = ["html"] * (close.end() - close.start())
                    i = close.end()
                else:
                    i = n
            else:
                i = j
            continue
        i += 1
    return lab


def token_sections(output: str, char_offsets: list[int], *, detail: bool = False) -> list[str]:
    from entropytrail.extract import app_span

    span = app_span(output)
    if span is None:
        return ["outside"] * len(char_offsets)
    s, e = span
    lab = char_sections(output[s:e], detail=detail)
    return [lab[o - s] if s <= o < e else "outside" for o in char_offsets]
