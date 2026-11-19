import random

import numpy as np

from entropytrail.model import generate_recorded
from entropytrail.teacher_forcing import compare, score_tokens


def test_matches_recorder(tiny_model, fake_io):
    rec = generate_recorded(
        tiny_model, [1, 2, 3], seed=3, temperature=1.0, max_tokens=200,
        eos_ids=set(), token_texts=fake_io["token_texts"],
    )
    assert len(rec.tokens) == 200
    scored = score_tokens(tiny_model, rec.prompt_ids, [t.token_id for t in rec.tokens])
    res = compare(rec, scored)
    print("COMPARE", res)
    assert res["ok"] is True


def test_chunk_size_invariance(tiny_model):
    rnd = random.Random(1)
    out = [rnd.randrange(1000) for _ in range(60)]
    a = score_tokens(tiny_model, [1, 2, 3], out, chunk=7)
    b = score_tokens(tiny_model, [1, 2, 3], out, chunk=512)
    np.testing.assert_allclose(a["H"], b["H"], atol=1e-4)


def test_long_sequence_bounded(tiny_model):
    rnd = random.Random(2)
    out = [rnd.randrange(1000) for _ in range(3000)]
    s = score_tokens(tiny_model, [1, 2, 3], out, chunk=256)
    assert len(s["H"]) == 3000
