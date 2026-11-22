import json
import time

import pytest

from entropytrail.render import render_app

pytestmark = pytest.mark.browser

OK = ('<!DOCTYPE html><html><body><button onclick="document.body.dataset.c=1">Go</button>'
      '<p>hello</p></body></html>')
BROKEN = "<html><body><p>x</p><script>undefinedFn()</script></body></html>"
HANG = "<html><body><script>while(true){}</script></body></html>"
EXTERNAL = ('<html><head><script src="https://example.com/x.js"></script></head>'
            "<body>x</body></html>")
RANDOM_APP = "<html><body><script>window.__r = Math.random()</script>r</body></html>"
ANIMATED = """<html><body><canvas id=c width=300 height=200></canvas><script>
const g=document.getElementById('c').getContext('2d');
setInterval(()=>{g.fillStyle='hsl('+Math.floor(Math.random()*360)+',70%,50%)';
g.fillRect(Math.random()*280, (Date.now()/7)%180, 20, 20);}, 30);
document.addEventListener('keydown',e=>{g.fillStyle='#000';g.fillRect(0,0,40,40);});
</script></body></html>"""


def mulberry32(seed: int):
    a = seed & 0xFFFFFFFF
    while True:
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        yield ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296


MULBERRY32_12345_FIRST = next(mulberry32(12345))


def test_ok_app_renders(browser, tmp_path):
    r = render_app(OK, tmp_path, browser=browser)
    assert r["renders"] and r["n_page_errors"] == 0 and r["body_text_chars"] >= 5
    assert (tmp_path / "load.png").stat().st_size > 0 and (tmp_path / "interact.png").exists()
    assert json.loads((tmp_path / "render.json").read_text())["renders"] is True


def test_uncaught_error_means_not_rendering(browser, tmp_path):
    r = render_app(BROKEN, tmp_path, browser=browser)
    assert not r["renders"] and r["n_page_errors"] >= 1


def test_hanging_app_times_out(browser, tmp_path):
    t = time.monotonic()
    r = render_app(HANG, tmp_path, browser=browser, timeout_s=3)
    assert r["timeout"] and not r["renders"] and time.monotonic() - t < 20
    assert (tmp_path / "load.png").exists() and (tmp_path / "interact.png").exists()


def test_browser_usable_after_hang(browser, tmp_path):
    render_app(HANG, tmp_path / "h", browser=browser, timeout_s=3)
    assert render_app(OK, tmp_path / "ok", browser=browser)["renders"]


def test_no_html_gives_blank_outcome(browser, tmp_path):
    r = render_app(None, tmp_path, browser=browser)
    assert not r["renders"] and (tmp_path / "interact.png").exists()


def test_external_requests_blocked(browser, tmp_path):
    r = render_app(EXTERNAL, tmp_path, browser=browser)
    assert r["blocked_requests"] >= 1


def test_random_is_seeded(browser, tmp_path):
    r = render_app(RANDOM_APP, tmp_path, browser=browser)
    assert r["first_random"] == pytest.approx(MULBERRY32_12345_FIRST)


def test_animated_random_app_renders_identically_twice(browser, tmp_path):
    render_app(ANIMATED, tmp_path / "a", browser=browser)
    render_app(ANIMATED, tmp_path / "b", browser=browser)
    for f in ("load.png", "interact.png"):
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes()

WAAPI = """<html><body><div id=d style="width:50px;height:50px;background:red"></div><button>x</button>
<script>document.querySelector('button').onclick=()=>{for(let i=0;i<20;i++){const e=document.createElement('div');
e.style.cssText='position:absolute;width:9px;height:9px;background:blue;left:'+(Math.random()*500)+'px;top:200px';
document.body.appendChild(e);e.animate([{transform:'translateY(0)'},{transform:'translateY(300px)'}],{duration:800})}};
document.getElementById('d').animate([{opacity:1},{opacity:0}],{duration:2000,iterations:Infinity});</script></body></html>"""

COVERED = """<html><body><button id=b>Start</button>
<div style="position:fixed;inset:0;background:rgba(0,0,0,.5)"></div><p>covered</p></body></html>"""


def test_web_animations_render_identically_twice(browser, tmp_path):
    render_app(WAAPI, tmp_path / "a", browser=browser)
    render_app(WAAPI, tmp_path / "b", browser=browser)
    for f in ("load.png", "interact.png"):
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes()


def test_blocked_click_is_not_a_timeout(browser, tmp_path):
    r = render_app(COVERED, tmp_path, browser=browser)
    assert r["renders"] and not r["timeout"]
    assert r["interaction_failed"] == ["click"]
    assert (tmp_path / "interact.png").stat().st_size > 0
