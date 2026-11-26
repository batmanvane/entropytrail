from entropytrail.extract import app_span
from entropytrail.sections import CODE_SECTIONS, char_sections, token_sections


def test_detail_labels():
    h = "<style>/*a*/</style><!--b--><script>//c\n</script>"
    lab = char_sections(h, detail=True)
    assert [lab[h.index(x)] for x in ("/*a", "<!--", "//c")] == ["css-comment", "html-comment", "js-comment"]


def test_js_line_and_block_comments():
    h = "<script>let a=1; // hi\n/* b */ x()</script>"
    lab = char_sections(h)
    assert lab[h.index("// hi")] == "comment" and lab[h.index("hi")] == "comment"
    assert lab[h.index("/* b")] == "comment" and lab[h.index("x()")] == "js"
    assert lab[h.index("let")] == "js" and lab[0] == "html"


def test_markers_in_strings_are_code():
    h = "<script>u='http://x'; s=\"/* no */\"; t=`// no`;</script>"
    lab = char_sections(h)
    for frag in ("http://x", "/* no */", "// no"):
        assert lab[h.index(frag)] == "js"


def test_escaped_quote_in_string():
    h = "<script>s='it\\'s // x'; y=1 // real</script>"
    lab = char_sections(h)
    assert lab[h.index("// x")] == "js" and lab[h.index("// real")] == "comment"


def test_css_and_html_comments_and_text():
    h = "<style>a{color:red}/* c */</style><!-- note --><p>Hi</p>"
    lab = char_sections(h)
    assert lab[h.index("color")] == "css" and lab[h.index("/* c")] == "comment"
    assert lab[h.index("<!--")] == "comment" and lab[h.index("Hi")] == "text"
    assert lab[h.index("<p>")] == "html"


def test_css_double_slash_is_not_comment():
    h = "<style>b{background:url(http://x/y)}</style>"
    assert char_sections(h)[h.index("//x")] == "css"


def test_script_close_tag_is_html_and_length_matches():
    h = "<SCRIPT type='x'>a</SCRIPT>tail"
    lab = char_sections(h)
    assert len(lab) == len(h)
    assert lab[h.index("a<")] == "js" and lab[h.index("</SCRIPT")] == "html" and lab[h.index("tail")] == "text"


def test_token_sections_outside_and_first_char():
    out = "Sure:\n```html\n<!DOCTYPE html><script>// c\nx</script>\n```"
    s, e = app_span(out)
    offs = [0, s, out.index("// c"), out.index("x</"), len(out)]
    assert token_sections(out, offs) == ["outside", "html", "comment", "js", "outside"]
    assert token_sections(out, offs, detail=True)[2] == "js-comment"


def test_token_sections_no_app():
    assert token_sections("no app", [0, 3]) == ["outside", "outside"]


def test_app_span_none_without_html():
    assert app_span("no app here") is None


def test_app_span_matches_extracted():
    out = "x\n```html\n<html><p>a</p></html>\n```\n"
    s, e = app_span(out)
    assert out[s:e] == "<html><p>a</p></html>"


def test_code_sections():
    assert CODE_SECTIONS == frozenset({"html", "css", "js", "text"})


def test_quote_inside_regex_literal_does_not_open_a_string():
    h = "<script>s=s.replace(/\"/g,'&quot;'); let x=1; // real comment\n</script>"
    lab = char_sections(h)
    assert lab[h.index("// real")] == "comment" and lab[h.index("let x")] == "js"


def test_regex_with_slashes_and_class():
    h = "<script>r=/[/]+\\/x/g; return /a\\/b/.test(s) // c\n</script>"
    lab = char_sections(h)
    assert lab[h.index("// c")] == "comment" and lab[h.index("+\\/x")] == "js"


def test_division_is_not_a_regex():
    h = "<script>a = b / c; d = e / f // comment\n</script>"
    assert char_sections(h)[h.index("// comment")] == "comment"


def test_css_string_with_comment_marker():
    h = '<style>a::before{content:"/*"} b{color:red}</style>'
    lab = char_sections(h)
    assert lab[h.index("/*")] == "css" and lab[h.index("color")] == "css"


def test_unterminated_string_ends_at_newline():
    h = "<script>s='oops\nx=1 // c\n</script>"
    assert char_sections(h)[h.index("// c")] == "comment"
