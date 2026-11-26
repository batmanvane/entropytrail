"""Teacher-forced rescoring of finished grid units (D-042, exploratory).

The recorder keeps H, S and the top-20 log-probabilities of the *untempered* model
distribution. The rescoring feeds each unit's prompt and output back through the model once
and stores, per output position, what the recorder did not keep:

- ``logp2_eos``: log2 P(stop) under the model distribution (all end-of-sequence ids summed),
- ``logp2_eos_T``: the same under the sampling distribution at the condition temperature,
- ``H_T``: entropy (bits) of the sampling distribution at the condition temperature.

``H`` of the forced pass is compared with the recorded ``H`` as a check (chunked forcing
noise, D-021). Output per unit: ``rescore.parquet`` and ``rescore.json`` (written last).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from entropytrail.storage import is_done, read_tokens
from entropytrail.teacher_forcing import forced_logprobs

_LN2 = math.log(2.0)
VERSION = 1


def _stop_stats(lp: mx.array, eos: mx.array, temperature: float) -> dict[str, mx.array]:
    """lp: (N, V) natural log-probs at T = 1; eos: (k,) end-of-sequence ids."""
    lp = lp.astype(mx.float32)
    p = mx.exp(lp)
    H = -mx.sum(mx.where(p > 0, p * lp, 0.0), axis=-1) / _LN2
    eos_lp = mx.logsumexp(mx.take(lp, eos, axis=-1), axis=-1)
    lt = lp / temperature
    lt = lt - mx.logsumexp(lt, axis=-1, keepdims=True)
    pt = mx.exp(lt)
    H_T = -mx.sum(mx.where(pt > 0, pt * lt, 0.0), axis=-1) / _LN2
    eos_lt = mx.logsumexp(mx.take(lt, eos, axis=-1), axis=-1)
    return {"H": H, "H_T": H_T, "logp2_eos": eos_lp / _LN2, "logp2_eos_T": eos_lt / _LN2}


def rescore(model, prompt_ids: list[int], output_ids: list[int], eos_ids: set[int],
            temperature: float, chunk: int = 512) -> dict[str, np.ndarray]:
    eos = mx.array(sorted(eos_ids), dtype=mx.int32)
    parts = []
    for _ids, lp in forced_logprobs(model, prompt_ids, output_ids, chunk):
        st = _stop_stats(lp, eos, temperature)
        mx.eval(st)
        parts.append({k: np.array(v, dtype=np.float32) for k, v in st.items()})
    keys = ("H", "H_T", "logp2_eos", "logp2_eos_T")
    if not parts:
        return {k: np.zeros(0, dtype=np.float32) for k in keys}
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def grid_units(cond_dir: Path, prompt_ids: list[str] | None = None) -> list[Path]:
    units = sorted(p.parent for p in Path(cond_dir).glob("*/[0-9][0-9][0-9]/meta.json"))
    if prompt_ids:
        units = [u for u in units if u.parent.name in prompt_ids]
    return [u for u in units if is_done(u)]


def is_rescored(unit: Path) -> bool:
    try:
        return json.loads((Path(unit) / "rescore.json").read_text()).get("status") == "done"
    except (OSError, ValueError):
        return False


def rescore_unit(unit: Path, model, prompt_ids: list[int], eos_ids: set[int], temperature: float,
                 meta: dict, chunk: int = 512) -> dict:
    unit = Path(unit)
    rec = read_tokens(unit).select(["pos", "token_id", "H"]).to_pandas()
    out = rescore(model, prompt_ids, rec["token_id"].tolist(), eos_ids, temperature, chunk)
    dH = np.abs(out["H"].astype(np.float64) - rec["H"].to_numpy(np.float64))
    tmp = unit / "rescore.parquet.tmp"
    pq.write_table(pa.table({"pos": pa.array(rec["pos"].to_numpy(np.int32)),
                             **{k: pa.array(out[k]) for k in ("H_T", "logp2_eos", "logp2_eos_T")}}), tmp)
    os.replace(tmp, unit / "rescore.parquet")
    res = dict(meta, version=VERSION, temperature=temperature, chunk=chunk, eos_ids=sorted(eos_ids),
               n_tokens=len(rec), p99_abs_dH=float(np.percentile(dH, 99)) if len(dH) else 0.0,
               max_abs_dH=float(dH.max()) if len(dH) else 0.0, status="done")
    tmp = unit / "rescore.json.tmp"
    tmp.write_text(json.dumps(res, indent=2))
    os.replace(tmp, unit / "rescore.json")
    return res
