import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def img_emb():
    from entropytrail.embed import ImageEmbedder
    return ImageEmbedder()


@pytest.fixture(scope="module")
def code_emb():
    from entropytrail.embed import CodeEmbedder
    return CodeEmbedder()


def test_image_embeddings(img_emb, tmp_path):
    blank, red = tmp_path / "b.png", tmp_path / "r.png"
    Image.new("RGB", (1024, 768), "white").save(blank)
    img = Image.new("RGB", (1024, 768), "white")
    img.paste((200, 30, 30), (100, 100, 700, 500))
    img.save(red)
    E = img_emb.embed([blank, blank, red])
    assert E.shape[0] == 3 and E.dtype == np.float32
    assert np.allclose(np.linalg.norm(E, axis=1), 1, atol=1e-5)
    assert E[0] @ E[1] == pytest.approx(1.0, abs=1e-5)
    assert E[0] @ E[2] < 0.95


def test_code_embeddings(code_emb):
    a = "<html><body><canvas></canvas><script>function draw(){}</script></body></html>"
    b = "<html><body><h1>Quiz</h1><button>Next</button></body></html>"
    E = code_emb.embed([a, a, b])
    assert np.allclose(np.linalg.norm(E, axis=1), 1, atol=1e-5)
    assert E[0] @ E[1] == pytest.approx(1.0, abs=1e-4) and E[0] @ E[2] < 0.99
    assert code_emb.last_truncated == [False, False, False]
