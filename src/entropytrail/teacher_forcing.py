"""Teacher-forced recomputation of next-token statistics (validation of the recorder)."""
import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache

from entropytrail.stats import token_stats_batch
from entropytrail.storage import Recording

PREFILL_STEP = 2048


def _logprobs(logits: mx.array) -> mx.array:
    # Same op as mlx_lm generate_step, in the dtype of the logits (float32 when the
    # condition uses the float32 head, otherwise the model dtype).
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def score_tokens(model, prompt_ids: list[int], output_ids: list[int],
                 chunk: int = 512) -> dict[str, np.ndarray]:
    """Stats of the distribution that produced each output token, in one forced pass."""
    cache = make_prompt_cache(model)
    n = len(output_ids)
    if n == 0:
        return token_stats_batch(mx.zeros((0, 1)), mx.zeros((0,), dtype=mx.int32))
    prompt = mx.array(prompt_ids, dtype=mx.int32)
    # Prefill all but the last prompt token in steps, then include it with the first chunk.
    while prompt.size > 1:
        k = min(PREFILL_STEP, prompt.size - 1)
        model(prompt[:k][None], cache=cache)
        mx.eval([c.state for c in cache])
        prompt = prompt[k:]
    out = mx.array(output_ids, dtype=mx.int32)
    pieces: list[dict[str, np.ndarray]] = []
    carry = prompt  # last prompt token, predicts output[0]
    for s in range(0, n, chunk):
        ids = out[s:s + chunk]
        # Inputs: carry + ids[:-1]; the last id of this chunk is fed with the next chunk.
        x = mx.concatenate([carry, ids[:-1]]) if ids.size > 1 else carry
        carry = ids[-1:]
        logits = model(x[None], cache=cache)[0, -ids.size:, :]
        lp = _logprobs(logits)
        pieces.append(token_stats_batch(lp, ids))
        mx.eval([c.state for c in cache])
    return {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}


def compare(rec: Recording, scored: dict, *, stat: str = "p99", tol: float = 0.01) -> dict:
    h = np.array([t.stats.H for t in rec.tokens], dtype=np.float64)
    s = np.array([t.stats.S for t in rec.tokens], dtype=np.float64)
    dh = np.abs(h - scored["H"].astype(np.float64))
    ds = np.abs(s - scored["S"].astype(np.float64))
    p99h, p99s = float(np.percentile(dh, 99)), float(np.percentile(ds, 99))
    mxh, mxs = float(dh.max()), float(ds.max())
    if stat == "max":
        ok = mxh <= tol and mxs <= tol
    else:
        ok = p99h <= tol and p99s <= tol
    return {
        "p99_abs_dH": p99h, "p99_abs_dS": p99s,
        "max_abs_dH": mxh, "max_abs_dS": mxs,
        "ok": bool(ok),
    }


def validate_recording(model, rec: Recording, chunk: int = 512) -> dict:
    """Chunk=1 bookkeeping check (max <= 1e-5 bits, decides ``ok``) plus informational chunk=N noise."""
    ids = [t.token_id for t in rec.tokens]
    bookkeeping = compare(rec, score_tokens(model, rec.prompt_ids, ids, chunk=1),
                          stat="max", tol=1e-5)
    noise = compare(rec, score_tokens(model, rec.prompt_ids, ids, chunk=chunk))
    noise.pop("ok")  # informational only (decision D-021)
    noise["chunk"] = chunk
    return {"bookkeeping_chunk1": bookkeeping, "chunked_noise": noise,
            "ok": bool(bookkeeping["ok"])}
