"""Fork experiments (spec §5.4, WP5): site selection, plan file, rollouts.

A fork plan fixes, per app, the fork sites (10 top-H code tokens + 3 random code tokens,
at least ``min_spacing`` tokens apart) and five variants per site: the top-3 alternatives
that differ from the actual token (``alt1..alt3``, one rollout each) and two re-samples
with the actual token (``same1``, ``same2``, baseline).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from entropytrail.sections import CODE_SECTIONS, token_sections
from entropytrail.storage import is_done, read_tokens

VARIANTS = ("alt1", "alt2", "alt3", "same1", "same2")
PARAMS = {"n_top": 10, "n_random": 3, "min_spacing": 20, "n_alt": 3, "n_same": 2}
PLAN_VERSION = 1


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "big")


def select_sites(H, sections: list[str], *, n_top: int = 10, n_random: int = 3,
                 min_spacing: int = 20, seed: int) -> list[tuple[int, str]]:
    H = np.asarray(H, dtype=np.float64)
    n = len(H)
    cand = [i for i in range(n - 1) if sections[i] in CODE_SECTIONS]
    chosen: list[tuple[int, str]] = []

    def free(i: int) -> bool:
        return all(abs(i - p) >= min_spacing for p, _ in chosen)

    for i in sorted(cand, key=lambda i: (-H[i], i)):
        if len(chosen) >= n_top:
            break
        if free(i):
            chosen.append((i, "top"))
    rng = np.random.default_rng(seed)
    pool = [i for i in cand if free(i)]
    while pool and sum(k == "random" for _, k in chosen) < n_random:
        i = int(pool[rng.integers(len(pool))])
        chosen.append((i, "random"))
        pool = [j for j in pool if free(j)]
    return sorted(chosen)


def alternatives(topk_ids, actual_id: int, n: int = 3) -> tuple[int, ...]:
    return tuple(int(t) for t in topk_ids if int(t) != int(actual_id))[:n]


def choose_apps(cond_dir: Path, prompt_ids: list[str], n_per_prompt: int, seed: int) -> list[tuple[str, int]]:
    out = []
    for pid in prompt_ids:
        ok = []
        for u in sorted((Path(cond_dir) / pid).iterdir()):
            if not (u.name.isdigit() and len(u.name) == 3 and is_done(u)):
                continue
            if json.loads((u / "meta.json").read_text()).get("html_extracted"):
                ok.append(int(u.name))
        rng = np.random.default_rng(_seed(seed, pid))
        pick = rng.choice(len(ok), size=min(n_per_prompt, len(ok)), replace=False)
        out += [(pid, ok[i]) for i in pick]
    return sorted(out)


def _app_sites(unit: Path, seed: int, params: dict) -> dict:
    t = read_tokens(unit).to_pydict()
    out = (unit / "output.txt").read_text(encoding="utf-8")
    sec = token_sections(out, t["char_offset"])
    H = np.asarray(t["H"], np.float64)
    app_idx = [i for i, s in enumerate(sec) if s != "outside"]
    first, n_app = (app_idx[0], len(app_idx)) if app_idx else (0, 1)
    sites = []
    for pos, kind in select_sites(H, sec, n_top=params["n_top"], n_random=params["n_random"],
                                  min_spacing=params["min_spacing"], seed=seed):
        actual = int(t["token_id"][pos])
        alts = alternatives(t["topk_ids"][pos], actual, params["n_alt"])
        variants = {f"alt{k + 1}": a for k, a in enumerate(alts)}
        variants.update({f"same{k + 1}": actual for k in range(params["n_same"])})
        sites.append({
            "pos": pos, "kind": kind, "H": float(H[pos]), "S": float(t["S"][pos]),
            "section": sec[pos], "rel_pos": (pos - first) / n_app, "actual_id": actual,
            "variants": variants,
        })
    return {"n_tokens": len(H), "sites": sites}


def build_plan(cond_dir: Path, apps: list[tuple[str, int]], *, seed: int, params: dict | None = None) -> dict:
    cond_dir = Path(cond_dir)
    params = dict(params or PARAMS)
    cid = json.loads((cond_dir / "condition.json").read_text())["condition_id"]
    plan_apps = []
    for pid, sample in apps:
        unit = cond_dir / pid / f"{sample:03d}"
        plan_apps.append({"prompt_id": pid, "sample": sample,
                          **_app_sites(unit, _seed(seed, pid, sample), params)})
    return {"plan_version": PLAN_VERSION, "condition_id": cid,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "seed": seed, "params": params, "apps": plan_apps}


def write_plan(path: Path, plan: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(plan, indent=1), encoding="utf-8")


def read_plan(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
