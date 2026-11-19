"""Experimental conditions, prompt sets and deterministic seed derivation."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Condition:
    name: str
    model_repo: str
    precision: str
    runtime: str
    runtime_version: str
    temperature: float
    max_tokens: int
    thinking: bool
    system_prompt_version: str
    machine: str

    @property
    def condition_id(self) -> str:
        canonical = json.dumps(
            dataclasses.asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"{self.name}-{digest[:8]}"


def load_condition(path: Path) -> Condition:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Condition(**data)


@dataclass(frozen=True)
class Prompt:
    id: str
    domain: str
    openness: str
    text: str


@dataclass(frozen=True)
class PromptSet:
    version: str
    system_prompts: dict[str, str]
    prompts: list[Prompt]


def load_prompts(path: Path) -> PromptSet:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    prompts = [Prompt(**p) for p in data["prompts"]]
    ids = [p.id for p in prompts]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate prompt ids: {dupes}")
    return PromptSet(
        version=data["version"],
        system_prompts={k: str(v) for k, v in data["system_prompts"].items()},
        prompts=prompts,
    )


def derive_seed(condition_id: str, prompt_id: str, sample: int) -> int:
    digest = hashlib.sha256(f"{condition_id}|{prompt_id}|{sample}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")
