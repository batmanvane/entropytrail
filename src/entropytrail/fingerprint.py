"""Structural fingerprint of an app: DOM tags, JS function names, CSS colours (spec §5.5)."""
from __future__ import annotations

import math
import re
from collections import Counter
from html.parser import HTMLParser

_SCRIPT = re.compile(r"<script\b[^>]*>(.*?)</script>", re.S | re.I)
_STYLE = re.compile(r"<style\b[^>]*>(.*?)</style>", re.S | re.I)
_STYLE_ATTR = re.compile(r"""\bstyle\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.I)
_KEYWORDS = {"if", "for", "while", "switch", "catch", "function", "return", "with", "else"}
_FUNC = [
    re.compile(r"\bfunction\s*\*?\s*([A-Za-z_$][\w$]*)\s*\("),
    re.compile(r"\b([A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?function\b"),
    re.compile(r"\b([A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?(?:\([^()]*\)|[A-Za-z_$][\w$]*)\s*=>"),
    # class / object method shorthand: name(args) {
    re.compile(r"(?:^|[;{}\n])\s*(?:async\s+|static\s+|get\s+|set\s+)*([A-Za-z_$][\w$]*)\s*\([^()]*\)\s*\{"),
]
_NAMED = {
    "black", "silver", "gray", "white", "maroon", "red", "purple", "fuchsia", "green",
    "lime", "olive", "yellow", "navy", "blue", "teal", "aqua", "orange",
}
_COLOR = re.compile(
    r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b|\b(?:rgba?|hsla?)\([^)]*\)|\b(" + "|".join(sorted(_NAMED)) + r")\b",
    re.I,
)


class _Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags: Counter = Counter()

    def handle_starttag(self, tag, attrs):
        self.tags[tag.lower()] += 1


def _norm_color(c: str) -> str:
    c = c.lower()
    if c.startswith("#") and len(c) == 4:
        return "#" + "".join(ch * 2 for ch in c[1:])
    return re.sub(r"\s+", "", c)


def fingerprint(html: str | None) -> dict:
    if html is None:
        return {"tags": {}, "js_functions": [], "css_colors": []}
    p = _Tags()
    try:
        p.feed(html)
        p.close()
    except Exception:  # noqa: BLE001 - malformed HTML: keep what was parsed
        pass
    js = "\n".join(_SCRIPT.findall(html))
    funcs = {m.group(1) for rx in _FUNC for m in rx.finditer(js)} - _KEYWORDS
    css = "\n".join(_STYLE.findall(html)) + "\n" + "\n".join(a or b for a, b in _STYLE_ATTR.findall(html))
    colors = {_norm_color(m.group(0)) for m in _COLOR.finditer(css)}
    return {"tags": dict(p.tags), "js_functions": sorted(funcs), "css_colors": sorted(colors)}


def _jaccard_distance(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return 1.0 - len(sa & sb) / len(sa | sb)


def _cosine_distance(a: dict, b: dict) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    if a == b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return min(1.0, max(0.0, 1.0 - dot / (na * nb)))


def fingerprint_distance(a: dict, b: dict) -> float:
    """Mean of tag-histogram cosine distance and Jaccard distances of functions and colours."""
    return (
        _cosine_distance(a["tags"], b["tags"])
        + _jaccard_distance(a["js_functions"], b["js_functions"])
        + _jaccard_distance(a["css_colors"], b["css_colors"])
    ) / 3.0
