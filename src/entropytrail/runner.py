"""Resumable execution of generation units with an optional deadline."""
from __future__ import annotations

import dataclasses
import json
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
    clean_partials(root / cid)
    eos_ids, token_texts = io if io is not None else tokenizer_io(tokenizer)
    if prompt_ids_fn is None:
        system = ps.system_prompts[cond.system_prompt_version]

        def prompt_ids_fn(p: Prompt) -> list[int]:
            return build_prompt_ids(tokenizer, system, p.text, cond.thinking)

    prov = provenance()
    done = skipped = aborted = 0

    def expired() -> bool:
        return deadline is not None and now() >= deadline

    for u in units:
        final = run_dir(root, cid, u.prompt.id, u.sample)
        if is_done(final):
            skipped += 1
            continue
        if expired():
            break
        seed = derive_seed(cid, u.prompt.id, u.sample)
        t_start = now()
        try:
            rec = generate_recorded(
                model, prompt_ids_fn(u.prompt), seed=seed,
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
            system_prompt_version=cond.system_prompt_version, sample=u.sample,
            seed=seed, weights_fingerprint=fingerprint,
            started_at=t_start.isoformat(), finished_at=now().isoformat(),
        )
        write_run(final, rec, meta)
        done += 1

    entry = {
        "started": started.isoformat(), "ended": now().isoformat(),
        "condition_id": cid, "done": done, "skipped": skipped, "aborted": aborted,
    }
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "runlog.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry
