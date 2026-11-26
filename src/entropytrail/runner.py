"""Resumable execution of generation units with an optional deadline."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from entropytrail.condition import Condition, Prompt, PromptSet, derive_seed
from entropytrail.model import (
    Interrupted, build_prompt_ids, generate_recorded, provenance, tokenizer_io,
)
from entropytrail.storage import clean_partials, is_done, run_dir, write_run


@dataclass
class Unit:
    prompt: Prompt
    sample: int


def plan_units(ps: PromptSet, n_samples: int, prompt_ids: list[str] | None) -> list[Unit]:
    if prompt_ids is None:
        prompts = list(ps.prompts)
    else:
        by_id = {p.id: p for p in ps.prompts}
        missing = [i for i in prompt_ids if i not in by_id]
        if missing:
            raise KeyError(f"unknown prompt ids: {missing}")
        prompts = [by_id[i] for i in prompt_ids]
    return [Unit(p, s) for p in prompts for s in range(n_samples)]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class IdentityMismatch(RuntimeError):
    """Stored data was produced under a different identity than the current run."""


# Keys of condition.json that must match on every later run of the condition.
MANIFEST_KEYS = (
    "condition", "condition_id", "weights_fingerprint", "system_prompt_version",
    "system_prompt_sha256", "mlx", "mlx_lm",
)


def _check_manifest(path: Path, manifest: dict) -> None:
    """Create ``condition.json`` or verify that it matches the current run."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return
    stored = json.loads(path.read_text(encoding="utf-8"))
    diff = {k: (stored.get(k), manifest[k]) for k in MANIFEST_KEYS if stored.get(k) != manifest[k]}
    if diff:
        raise IdentityMismatch(
            f"{path} does not match the current run (stored vs current): {diff}; "
            "refusing to add units to this condition"
        )


def check_condition(root: Path, cond: Condition, ps: PromptSet, fingerprint: str, prov: dict,
                    started: datetime) -> tuple[str, str]:
    """Create or verify ``<root>/<condition_id>/condition.json``; return (system prompt, its sha256)."""
    system = ps.system_prompts[cond.system_prompt_version]
    system_sha = _sha256(system)
    _check_manifest(Path(root) / cond.condition_id / "condition.json", {
        "condition": dataclasses.asdict(cond), "condition_id": cond.condition_id,
        "weights_fingerprint": fingerprint,
        "system_prompt_version": cond.system_prompt_version,
        "system_prompt_sha256": system_sha,
        "mlx": prov["mlx"], "mlx_lm": prov["mlx_lm"],
        "created_at": started.isoformat(),
    })
    return system, system_sha


def _check_done_unit(final: Path, prompt: Prompt) -> None:
    meta = json.loads((final / "meta.json").read_text(encoding="utf-8"))
    stored = meta.get("prompt_text_sha256")
    if stored is None and isinstance(meta.get("prompt"), dict) and "text" in meta["prompt"]:
        stored = _sha256(meta["prompt"]["text"])
    if stored != _sha256(prompt.text):
        raise IdentityMismatch(
            f"done unit {final} was generated from a different text of prompt "
            f"{prompt.id!r} (prompt_text_sha256 {stored} != {_sha256(prompt.text)})"
        )


def _move_stale(final: Path, now: datetime) -> Path:
    base = final.parent / f"{final.name}.stale-{now.strftime('%Y%m%d-%H%M%S')}"
    target, i = base, 1
    while target.exists():
        target = base.parent / f"{base.name}-{i}"
        i += 1
    os.rename(final, target)
    return target


def run_units(
    units: list[Unit],
    cond: Condition,
    ps: PromptSet,
    model,
    tokenizer,
    root: Path,
    deadline: datetime | None,
    *,
    fingerprint: str,
    now: Callable[[], datetime] = datetime.now,
    io: tuple | None = None,
    prompt_ids_fn: Callable[[Prompt], list[int]] | None = None,
) -> dict:
    root = Path(root)
    cid = cond.condition_id
    started = now()
    prov = provenance()
    done = skipped = aborted = 0
    stale_moved: list[str] = []
    error: str | None = None

    def expired() -> bool:
        return deadline is not None and now() >= deadline

    try:
        system, system_sha = check_condition(root, cond, ps, fingerprint, prov, started)
        clean_partials(root / cid)
        eos_ids, token_texts = io if io is not None else tokenizer_io(tokenizer)
        if prompt_ids_fn is None:
            def prompt_ids_fn(p: Prompt) -> list[int]:
                return build_prompt_ids(tokenizer, system, p.text, cond.thinking)

        for u in units:
            final = run_dir(root, cid, u.prompt.id, u.sample)
            if is_done(final):
                _check_done_unit(final, u.prompt)
                skipped += 1
                continue
            if expired():
                break
            if final.exists():
                stale_moved.append(str(_move_stale(final, now()).relative_to(root)))
            seed = derive_seed(cid, u.prompt.id, u.sample)
            t_start = now()
            pids = [int(i) for i in prompt_ids_fn(u.prompt)]
            try:
                rec = generate_recorded(
                    model, pids, seed=seed,
                    temperature=cond.temperature, max_tokens=cond.max_tokens,
                    eos_ids=eos_ids, token_texts=token_texts, should_stop=expired,
                )
            except Interrupted:
                shutil.rmtree(final.parent / (final.name + ".partial"), ignore_errors=True)
                aborted += 1
                break
            meta = dict(prov)
            meta.update(
                condition=dataclasses.asdict(cond), condition_id=cid,
                prompt=dataclasses.asdict(u.prompt), prompt_set_version=ps.version,
                system_prompt_version=cond.system_prompt_version,
                system_prompt_sha256=system_sha, prompt_text_sha256=_sha256(u.prompt.text),
                prompt_ids=pids, sample=u.sample,
                seed=seed, weights_fingerprint=fingerprint,
                started_at=t_start.isoformat(), finished_at=now().isoformat(),
            )
            write_run(final, rec, meta)
            done += 1
    except BaseException as e:
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        entry = {
            "started": started.isoformat(), "ended": now().isoformat(),
            "condition_id": cid, "done": done, "skipped": skipped, "aborted": aborted,
            "error": error, "git_commit": prov["git_commit"],
            "weights_fingerprint": fingerprint, "stale_moved": stale_moved,
        }
        root.mkdir(parents=True, exist_ok=True)
        with open(root / "runlog.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    return entry
