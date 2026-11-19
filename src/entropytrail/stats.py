"""Per-token information statistics (all quantities in bits).

Inputs are natural-log probability vectors as yielded by
``mlx_lm.generate.generate_step``; everything is computed in float32.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

TOPK = 20
_LN2 = math.log(2.0)


@dataclass(frozen=True)
class TokenStats:
    H: float
    S: float
    varentropy: float
    margin: float
    topk_ids: np.ndarray  # int32, (min(TOPK, V),), sorted by descending probability
    topk_logp2: np.ndarray  # float32, same shape, log2 probabilities


def _compute(logprobs: mx.array, token_ids: mx.array) -> dict[str, mx.array]:
    """Core computation for a (N, V) block and (N,) token ids."""
    lp2 = logprobs.astype(mx.float32) / _LN2  # nats -> bits
    p = mx.exp(lp2 * _LN2)
    pos = p > 0
    safe = mx.where(pos, lp2, 0.0)  # -inf entries contribute exactly 0
    H = -mx.sum(mx.where(pos, p * safe, 0.0), axis=-1)
    second = mx.sum(mx.where(pos, p * safe * safe, 0.0), axis=-1)
    varentropy = mx.maximum(second - H * H, 0.0)
    S = -mx.take_along_axis(lp2, token_ids.astype(mx.int32)[:, None], axis=-1)[:, 0]

    V = lp2.shape[-1]
    k = min(TOPK, V)
    if k < V:
        idx = mx.argpartition(-lp2, kth=k - 1, axis=-1)[..., :k]
    else:
        idx = mx.broadcast_to(mx.arange(V), lp2.shape)
    vals = mx.take_along_axis(lp2, idx, axis=-1)
    order = mx.argsort(-vals, axis=-1)
    topk_ids = mx.take_along_axis(idx, order, axis=-1).astype(mx.int32)
    topk_logp2 = mx.take_along_axis(vals, order, axis=-1)
    if k >= 2:
        margin = topk_logp2[:, 0] - topk_logp2[:, 1]
    else:
        margin = mx.zeros_like(H)
    return {
        "H": H, "S": S, "varentropy": varentropy, "margin": margin,
        "topk_ids": topk_ids, "topk_logp2": topk_logp2,
    }


def token_stats_batch(logprobs: mx.array, token_ids: mx.array) -> dict[str, np.ndarray]:
    """Statistics for a (T, V) block of log-probs and (T,) token ids."""
    out = _compute(logprobs, token_ids)
    mx.eval(out)
    return {
        "H": np.array(out["H"], dtype=np.float32),
        "S": np.array(out["S"], dtype=np.float32),
        "varentropy": np.array(out["varentropy"], dtype=np.float32),
        "margin": np.array(out["margin"], dtype=np.float32),
        "topk_ids": np.array(out["topk_ids"], dtype=np.int32),
        "topk_logp2": np.array(out["topk_logp2"], dtype=np.float32),
    }


def token_stats(logprobs: mx.array, token_id: int) -> TokenStats:
    """Statistics for one 1-D log-prob vector and the sampled token id."""
    b = token_stats_batch(logprobs[None, :], mx.array([int(token_id)]))
    return TokenStats(
        H=float(b["H"][0]),
        S=float(b["S"][0]),
        varentropy=float(b["varentropy"][0]),
        margin=float(b["margin"][0]),
        topk_ids=b["topk_ids"][0],
        topk_logp2=b["topk_logp2"][0],
    )
