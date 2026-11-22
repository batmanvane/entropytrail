"""Resumable automatic evaluation of units (WP4): render, fingerprint, embeddings.

Every done unit (a grid app ``<prompt>/NNN/`` or a fork rollout
``<prompt>/NNN/forks/PPPPP/<variant>/``) gets an ``eval/`` folder, written atomically
via ``eval.partial/``. See docs/DATA.md for the files and keys.
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from entropytrail.fingerprint import fingerprint
from entropytrail.outcome import Outcome, screen_embedding
from entropytrail.render import RENDER_VERSION
from entropytrail.storage import is_done

_SAMPLE = re.compile(r"\d{3}")
_POS = re.compile(r"\d{5}")
_VARIANT = re.compile(r"(alt|same)\d+")
BATCH = 16


def find_units(cond_dir: Path, *, prompt_ids: list[str] | None, forks: bool) -> list[Path]:
    cond_dir = Path(cond_dir)
    out: list[Path] = []
    for pdir in sorted(p for p in cond_dir.iterdir() if p.is_dir()):
        if pdir.name.startswith("smoke") or (prompt_ids is not None and pdir.name not in prompt_ids):
            continue
        for u in sorted(p for p in pdir.iterdir() if p.is_dir() and _SAMPLE.fullmatch(p.name)):
            if not is_done(u):
                continue
            out.append(u)
            if forks and (u / "forks").is_dir():
                for pos in sorted(p for p in (u / "forks").iterdir() if _POS.fullmatch(p.name)):
                    for v in sorted(p for p in pos.iterdir() if _VARIANT.fullmatch(p.name)):
                        if is_done(v):
                            out.append(v)
    return out


def _valid(unit: Path, image_model: str, code_model: str) -> bool:
    try:
        m = json.loads((unit / "eval" / "eval.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (m.get("status") == "done" and m.get("render_version") == RENDER_VERSION
            and m.get("image_model") == image_model and m.get("code_model") == code_model)


def _git_commit() -> str:
    from entropytrail.model import _git

    return _git("rev-parse", "HEAD") or "unknown"


def _code_text(unit: Path) -> str:
    """Text for the code embedding: the app, or the raw output if no HTML was extracted."""
    app = unit / "app.html"
    if app.exists():
        return app.read_text(encoding="utf-8")
    out = unit / "output.txt"
    return out.read_text(encoding="utf-8") if out.exists() else ""


def eval_units(units: list[Path], *, render_fn: Callable[[str | None, Path], dict],
               image_embedder, code_embedder, log: Callable[[str], None] = print) -> dict:
    counts = {"done": 0, "skipped": 0, "failed": 0}
    commit = _git_commit()
    pending: list[Path] = []

    def fail(unit: Path, err: BaseException) -> None:
        counts["failed"] += 1
        shutil.rmtree(unit / "eval.partial", ignore_errors=True)
        log(f"eval failed: {unit}: {type(err).__name__}: {err}")

    def flush() -> None:
        if not pending:
            return
        batch = list(pending)
        pending.clear()
        try:
            img = image_embedder.embed([p for u in batch for p in
                                        (u / "eval.partial" / "load.png", u / "eval.partial" / "interact.png")])
            code = code_embedder.embed([_code_text(u) for u in batch])
            truncated = list(getattr(code_embedder, "last_truncated", [False] * len(batch)))
        except Exception as e:  # noqa: BLE001
            for u in batch:
                fail(u, e)
            return
        for i, u in enumerate(batch):
            try:
                part = u / "eval.partial"
                il, ii = img[2 * i], img[2 * i + 1]
                np.savez(part / "embeddings.npz", img_load=il, img_interact=ii,
                         screen=screen_embedding(il, ii), code=code[i])
                (part / "eval.json").write_text(json.dumps({
                    "status": "done", "render_version": RENDER_VERSION,
                    "image_model": image_embedder.model_id, "code_model": code_embedder.model_id,
                    "code_truncated": bool(truncated[i]), "git_commit": commit,
                    "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                }, indent=2), encoding="utf-8")
                if (u / "eval").exists():
                    shutil.rmtree(u / "eval")
                part.rename(u / "eval")
                counts["done"] += 1
            except Exception as e:  # noqa: BLE001
                fail(u, e)

    for u in units:
        u = Path(u)
        if _valid(u, image_embedder.model_id, code_embedder.model_id):
            counts["skipped"] += 1
            continue
        part = u / "eval.partial"
        shutil.rmtree(part, ignore_errors=True)
        try:
            app = u / "app.html"
            html = app.read_text(encoding="utf-8") if app.exists() else None
            render_fn(html, part)
            (part / "fingerprint.json").write_text(json.dumps(fingerprint(html), indent=2), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            fail(u, e)
            continue
        pending.append(u)
        if len(pending) >= BATCH:
            flush()
    flush()
    return counts


def load_outcome(unit: Path) -> Outcome:
    ev = Path(unit) / "eval"
    r = json.loads((ev / "render.json").read_text(encoding="utf-8"))
    return Outcome(bool(r["renders"]), np.load(ev / "embeddings.npz")["screen"])
