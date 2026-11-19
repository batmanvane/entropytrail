"""Atomic on-disk storage of one generation run (see docs/DATA.md)."""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from entropytrail.extract import extract_html
from entropytrail.stats import TokenStats

DEFAULT_ROOT = "/Volumes/Data/entropytrail"


@dataclass
class TokenRecord:
    pos: int
    token_id: int
    text: str
    char_offset: int
    stats: TokenStats


@dataclass
class Recording:
    prompt_ids: list[int]
    tokens: list[TokenRecord]
    text: str
    finish_reason: Literal["eos", "max_tokens"]
    seconds: float


def data_root() -> Path:
    """Raw data root; never creates directories."""
    return Path(os.environ.get("ENTROPYTRAIL_DATA") or DEFAULT_ROOT)


def run_dir(root: Path, condition_id: str, prompt_id: str, sample: int) -> Path:
    return Path(root) / condition_id / prompt_id / f"{sample:03d}"


def _partial(final_dir: Path) -> Path:
    return final_dir.parent / (final_dir.name + ".partial")


def _table(rec: Recording) -> pa.Table:
    t = rec.tokens
    i32, f32 = pa.int32(), pa.float32()
    return pa.table(
        {
            "pos": pa.array([r.pos for r in t], i32),
            "token_id": pa.array([r.token_id for r in t], i32),
            "text": pa.array([r.text for r in t], pa.string()),
            "char_offset": pa.array([r.char_offset for r in t], i32),
            "H": pa.array([r.stats.H for r in t], f32),
            "S": pa.array([r.stats.S for r in t], f32),
            "varentropy": pa.array([r.stats.varentropy for r in t], f32),
            "margin": pa.array([r.stats.margin for r in t], f32),
            "topk_ids": pa.array(
                [np.asarray(r.stats.topk_ids, dtype=np.int32).tolist() for r in t],
                pa.list_(i32),
            ),
            "topk_logp2": pa.array(
                [np.asarray(r.stats.topk_logp2, dtype=np.float32).tolist() for r in t],
                pa.list_(f32),
            ),
        }
    )


def write_run(final_dir: Path, rec: Recording, meta: dict) -> None:
    final_dir = Path(final_dir)
    if final_dir.exists():
        raise FileExistsError(final_dir)
    part = _partial(final_dir)
    if part.exists():
        shutil.rmtree(part)
    part.mkdir(parents=True)
    try:
        pq.write_table(_table(rec), part / "tokens.parquet")
        (part / "output.txt").write_text(rec.text, encoding="utf-8")
        html = extract_html(rec.text)
        if html is not None:
            (part / "app.html").write_text(html, encoding="utf-8")
        n = len(rec.tokens)
        out = dict(meta)
        out.update(
            n_tokens=n,
            finish_reason=rec.finish_reason,
            seconds=rec.seconds,
            tokens_per_s=(n / rec.seconds) if rec.seconds else 0.0,
            html_extracted=html is not None,
            status="done",
        )
        (part / "meta.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.rename(part, final_dir)
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)
        raise


def is_done(final_dir: Path) -> bool:
    p = Path(final_dir) / "meta.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("status") == "done"
    except (OSError, ValueError):
        return False


def read_tokens(final_dir: Path) -> pa.Table:
    return pq.read_table(Path(final_dir) / "tokens.parquet")


def clean_partials(root: Path) -> int:
    root = Path(root)
    if not root.exists():
        return 0
    n = 0
    for p in sorted(root.rglob("*.partial")):
        if p.is_dir():
            shutil.rmtree(p)
            n += 1
    return n
