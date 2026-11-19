from entropytrail.extract import extract_html

DOC = "<!DOCTYPE html>\n<html><body><p>hi</p></body></html>"


def test_plain_document_unchanged_stripped():
    assert extract_html("\n  " + DOC + "\n\n") == DOC


def test_fenced_html_with_prose():
    text = f"Here you go:\n```html\n{DOC}\n```\nHope it helps!"
    assert extract_html(text) == DOC


def test_fence_without_language():
    assert extract_html(f"```\n{DOC}\n```") == DOC


def test_uppercase_language_tag():
    assert extract_html(f"x\n```HTML\n{DOC}\n```") == DOC


def test_first_qualifying_block_wins():
    text = (
        "```python\nprint('x')\n```\n"
        "```html\n<div>fragment</div>\n```\n"
        "```html\n<HTML><body>A</body></HTML>\n```\n"
        "```html\n<!doctype html><html>B</html>\n```\n"
    )
    assert extract_html(text) == "<HTML><body>A</body></HTML>"


def test_no_html_returns_none():
    assert extract_html("Sorry, I cannot do that.") is None
    assert extract_html("```html\n<div>x</div>\n```") is None
    assert extract_html("") is None


def test_unterminated_fence_truncated():
    text = "Sure:\n```html\n<!DOCTYPE html>\n<html><body>cut off"
    assert extract_html(text) == "<!DOCTYPE html>\n<html><body>cut off"


def test_fallback_slice_without_fence():
    text = "Intro text\n<html><body>x</body></html>\ntrailing chatter"
    assert extract_html(text) == "<html><body>x</body></html>"


def test_fallback_slice_without_closing_tag():
    assert extract_html("blah <!doctype html><html><body>x") == "<!doctype html><html><body>x"
