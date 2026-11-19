"""Model loading, provenance and the recording generation loop."""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Callable

import mlx.core as mx

from entropytrail.condition import Condition
from entropytrail.stats import token_stats
from entropytrail.storage import Recording, TokenRecord


class Interrupted(Exception):
    """Raised when should_stop() fires during generation."""


def snapshot_dir(repo: str) -> Path:
    """Newest local HF snapshot directory of ``repo`` (no network)."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, local_files_only=True))


def load_model(cond: Condition):
    import mlx.utils
    from mlx_lm import load

    model, tokenizer = load(cond.model_repo)
    if cond.precision == "bf16":
        emb = dict(mlx.utils.tree_flatten(model.parameters()))
        key = next(k for k in emb if k.endswith("embed_tokens.weight"))
        assert emb[key].dtype == mx.bfloat16, f"expected bf16 weights, got {emb[key].dtype}"
    if cond.logits_precision == "float32":
        _use_float32_head(model)
    elif cond.logits_precision != "native":
        raise ValueError(f"unknown logits_precision: {cond.logits_precision!r}")
    return model, tokenizer


def _use_float32_head(model) -> None:
    """Replace lm_head by a float32 copy so logits and log-softmax are float32."""
    import mlx.nn as nn

    lm = model.language_model
    if not hasattr(lm, "lm_head"):
        raise ValueError("model has tied embeddings / no lm_head; float32 head unsupported")

    class Float32Head(nn.Module):
        def __init__(self, weight: mx.array):
            super().__init__()
            self.weight = weight

        def __call__(self, x: mx.array) -> mx.array:
            return x.astype(mx.float32) @ self.weight.T

    lm.lm_head = Float32Head(lm.lm_head.weight.astype(mx.float32))
    mx.eval(lm.lm_head.weight)
    assert lm.lm_head.weight.dtype == mx.float32
    out = model(mx.array([[1]]))
    assert out.dtype == mx.float32, f"expected float32 logits, got {out.dtype}"


def weights_fingerprint(repo: str) -> str:
    """sha256 over sorted blob names (= content sha256) of the safetensors files."""
    snap = snapshot_dir(repo)
    names = sorted(os.path.basename(os.path.realpath(p)) for p in snap.glob("*.safetensors"))
    if not names:
        raise FileNotFoundError(f"no safetensors in {snap}")
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def _git(*args: str) -> str | None:
    root = Path(__file__).resolve().parents[2]
    try:
        r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def provenance() -> dict:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "git_commit": commit or "unknown",
        "git_dirty": None if commit is None or status is None else bool(status),
        "mlx": md.version("mlx"),
        "mlx_lm": md.version("mlx-lm"),
        "python": platform.python_version(),
        "macos": platform.mac_ver()[0],
        "hostname": socket.gethostname(),
    }


def build_prompt_ids(tokenizer, system: str, user: str, thinking: bool) -> list[int]:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=thinking
    )
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    return list(ids)


def tokenizer_io(tokenizer) -> tuple[set[int], Callable[[list[int]], list[str]]]:
    eos_ids = set(tokenizer.eos_token_ids)

    def token_texts(ids: list[int]) -> list[str]:
        det = tokenizer.detokenizer
        det.reset()
        out = []
        for i in ids:
            det.add_token(i)
            out.append(det.last_segment)
        det.finalize()
        tail = det.last_segment
        if tail:
            if out:
                out[-1] += tail
            else:
                out.append(tail)
        return out

    return eos_ids, token_texts


def generate_recorded(
    model,
    prompt_ids: list[int],
    *,
    seed: int,
    temperature: float,
    max_tokens: int,
    eos_ids: set[int],
    token_texts: Callable[[list[int]], list[str]],
    should_stop: Callable[[], bool] | None = None,
) -> Recording:
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    mx.random.seed(seed)
    t0 = time.monotonic()
    ids: list[int] = []
    stats = []
    finish = "max_tokens"
    for token, logprobs in generate_step(
        mx.array(prompt_ids), model, max_tokens=max_tokens,
        sampler=make_sampler(temp=temperature),
    ):
        if should_stop is not None and should_stop():
            raise Interrupted()
        token = int(token)
        ids.append(token)
        stats.append(token_stats(logprobs, token))
        if token in eos_ids:
            finish = "eos"
            break

    n_text = len(ids) - 1 if finish == "eos" else len(ids)
    texts = token_texts(ids[:n_text]) + [""] * (len(ids) - n_text)
    records, off = [], 0
    for pos, (tid, txt, st) in enumerate(zip(ids, texts, stats)):
        records.append(TokenRecord(pos=pos, token_id=tid, text=txt, char_offset=off, stats=st))
        off += len(txt)
    return Recording(
        prompt_ids=list(prompt_ids), tokens=records, text="".join(texts),
        finish_reason=finish, seconds=time.monotonic() - t0,
    )
