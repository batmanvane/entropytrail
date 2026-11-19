import math

import pytest

from entropytrail.model import (
    Interrupted, build_prompt_ids, generate_recorded, tokenizer_io,
    weights_fingerprint,
)

REPO = "Qwen/Qwen3.5-9B"


def run(model, fake_io, seed=7, max_tokens=50, **kw):
    args = dict(fake_io)
    args.update(kw)
    return generate_recorded(
        model, [1, 2, 3], seed=seed, temperature=1.0, max_tokens=max_tokens, **args
    )


def test_lengths_and_ranges(tiny_model, fake_io):
    rec = run(tiny_model, fake_io)
    assert 0 < len(rec.tokens) <= 50
    assert rec.prompt_ids == [1, 2, 3]
    off = 0
    for i, t in enumerate(rec.tokens):
        assert t.pos == i
        assert t.char_offset == off
        off += len(t.text)
        assert -1e-4 <= t.stats.H <= math.log2(1000) + 1e-4
        assert t.stats.S >= 0
    assert rec.text == "".join(t.text for t in rec.tokens)


def test_surprisal_consistent_with_topk(tiny_model, fake_io):
    rec = run(tiny_model, fake_io)
    hits = 0
    for t in rec.tokens:
        ids = list(t.stats.topk_ids)
        if t.token_id in ids:
            hits += 1
            assert t.stats.S == pytest.approx(-t.stats.topk_logp2[ids.index(t.token_id)], abs=1e-4)
    assert hits > 0


def test_same_seed_same_tokens(tiny_model, fake_io):
    a = [t.token_id for t in run(tiny_model, fake_io, seed=7).tokens]
    b = [t.token_id for t in run(tiny_model, fake_io, seed=7).tokens]
    c = [t.token_id for t in run(tiny_model, fake_io, seed=8).tokens]
    assert a == b
    assert a != c


def test_should_stop_raises(tiny_model, fake_io):
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] >= 5

    with pytest.raises(Interrupted):
        run(tiny_model, fake_io, should_stop=stop)


def test_max_tokens_flag(tiny_model, fake_io):
    rec = run(tiny_model, {**fake_io, "eos_ids": set()}, max_tokens=10)
    assert rec.finish_reason == "max_tokens"
    assert len(rec.tokens) == 10


def test_eos_recorded_without_text(tiny_model, fake_io):
    first = run(tiny_model, fake_io, max_tokens=3).tokens[1].token_id
    rec = run(tiny_model, {**fake_io, "eos_ids": {first}}, max_tokens=10)
    assert rec.finish_reason == "eos"
    assert len(rec.tokens) == 2
    assert rec.tokens[-1].token_id == first
    assert rec.tokens[-1].text == ""


def _tokenizer():
    from mlx_lm.utils import load_tokenizer
    from entropytrail.model import snapshot_dir

    return load_tokenizer(snapshot_dir(REPO))


@pytest.mark.slow
def test_real_tokenizer_chat_template():
    tok = _tokenizer()
    ids = build_prompt_ids(tok, "You are helpful.", "Build a clock.", thinking=False)
    text = tok.decode(ids)
    tail = text.split("Build a clock.")[1]
    assert tail.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


@pytest.mark.slow
def test_real_tokenizer_token_texts_roundtrip():
    tok = _tokenizer()
    eos, token_texts = tokenizer_io(tok)
    assert eos
    ids = tok.encode("Hallo Welt! Größe ü ä ö 😀 <div>x</div>\n\n  ok")
    assert "".join(token_texts(ids)) == tok.decode(ids)
    assert len(token_texts(ids)) == len(ids)


@pytest.mark.slow
def test_weights_fingerprint_stable():
    a = weights_fingerprint(REPO)
    assert a == weights_fingerprint(REPO)
    assert len(a) == 64 and all(c in "0123456789abcdef" for c in a)


def test_provenance_keys():
    from entropytrail.model import provenance

    p = provenance()
    for k in ("git_commit", "git_dirty", "mlx", "mlx_lm", "python", "macos", "hostname"):
        assert k in p
