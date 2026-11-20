import importlib.util
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("serve_apps", ROOT / "scripts" / "serve_apps.py")
serve_apps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve_apps)


def test_inject_puts_shim_first_in_head():
    out = serve_apps.inject("<html><head><script>alert(1)</script></head></html>")
    assert out.index("rating shim") < out.index("alert(1)")
    assert serve_apps.inject("<p>x</p>").startswith("<script>")


def test_serves_only_app_html(tmp_path):
    unit = tmp_path / "edu-tight-01" / "000"
    unit.mkdir(parents=True)
    (unit / "app.html").write_text("<html><head></head><body>ok</body></html>")
    (unit / "meta.json").write_text("{}")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), serve_apps.make_handler(tmp_path))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        body = urllib.request.urlopen(base + "/edu-tight-01/000/").read().decode()
        assert "ok" in body and "rating shim" in body
        for path in ("/edu-tight-01/000/meta.json", "/edu-tight-01/000/tokens.parquet", "/../etc/passwd"):
            try:
                urllib.request.urlopen(base + path)
                raise AssertionError(path)
            except urllib.error.HTTPError as e:
                assert e.code == 404
    finally:
        srv.shutdown()


def test_storage_guard_is_unit_specific():
    out = serve_apps.inject("<html><head></head></html>", "edu-tight-01/003")
    assert "__et_unit" in out and "edu-tight-01/003" in out
    assert out.index("rating shim") < out.index("storage guard")
