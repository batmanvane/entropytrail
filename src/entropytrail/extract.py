"""Extract a self-contained HTML document from raw LLM output."""

import re

_MARKER = re.compile(r"<html|<!doctype", re.IGNORECASE)
_FENCED = re.compile(r"```[ \t]*([\w+-]*)[^\n]*\n(.*?)```", re.IGNORECASE | re.DOTALL)
_OPEN_FENCE = re.compile(r"```[ \t]*(?:html)?[ \t]*\r?\n", re.IGNORECASE)


def extract_html(text: str) -> str | None:
    """Return the HTML document in ``text``, or None if there is none."""
    stripped = text.strip()
    if re.match(r"<html|<!doctype", stripped, re.IGNORECASE):
        return stripped

    for m in _FENCED.finditer(text):
        if m.group(1).lower() not in ("", "html"):
            continue
        body = m.group(2).strip()
        if _MARKER.search(body):
            return body

    # Truncated run: opening fence without a closing one.
    opening = _OPEN_FENCE.search(text)
    if opening:
        body = text[opening.end():].strip()
        if _MARKER.search(body) and "```" not in body:
            return body

    # Fallback: slice from first marker to last </html> (or end).
    start = _MARKER.search(text)
    if not start:
        return None
    end = text.lower().rfind("</html>")
    if end >= start.start():
        return text[start.start():end + len("</html>")].strip()
    return text[start.start():].strip()


def app_span(output: str) -> tuple[int, int] | None:
    """Character span of the extracted HTML document inside ``output``, or None."""
    html = extract_html(output)
    if html is None:
        return None
    start = output.find(html)
    if start < 0:
        return None
    return start, start + len(html)
