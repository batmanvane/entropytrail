from entropytrail.fingerprint import fingerprint, fingerprint_distance

H = ("<html><style>a{color:#FFF} b{background:rgb(0, 0, 0)} i{color:Red}</style><body><div><p>x</p></div>"
     "<span style='color:#00ff00'>s</span>"
     "<script>function draw(){} const step=()=>{}; let go = function(){}; "
     "class A { update(dt) { if (x) {} } } const later = async (a) => a; "
     "if (a) { for (;;) {} }</script></body></html>")


def test_fingerprint_fields():
    f = fingerprint(H)
    assert f["tags"]["div"] == 1 and f["tags"]["script"] == 1
    assert f["js_functions"] == ["draw", "go", "later", "step", "update"]
    assert f["css_colors"] == ["#00ff00", "#ffffff", "red", "rgb(0,0,0)"]


def test_fingerprint_none_is_empty():
    assert fingerprint(None) == {"tags": {}, "js_functions": [], "css_colors": []}


def test_distance_identity_and_bounds():
    f = fingerprint(H)
    assert fingerprint_distance(f, f) == 0.0
    assert fingerprint_distance(f, fingerprint(None)) == 1.0
    g = fingerprint(H.replace("draw", "paint"))
    assert 0.0 < fingerprint_distance(f, g) < 1.0


def test_both_empty_is_zero():
    e = fingerprint(None)
    assert fingerprint_distance(e, e) == 0.0
