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


# --- rollouts -------------------------------------------------------------------------

def rollout_dir(unit_dir: Path, pos: int, variant: str) -> Path:
    return Path(unit_dir) / "forks" / f"{pos:05d}" / variant


def _rollout_seed(cid: str, pid: str, sample: int, pos: int, variant: str) -> int:
    from entropytrail.condition import derive_seed

    return derive_seed(cid, f"{pid}/{sample:03d}/{pos:05d}/{variant}", 0)


def _check_done_rollout(d: Path, expect: dict) -> None:
    from entropytrail.runner import IdentityMismatch

    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    f = meta.get("fork", {})
    got = {"parent": f.get("parent"), "pos": f.get("pos"), "variant": f.get("variant"),
           "forced_token_id": f.get("forced_token_id"), "seed": meta.get("seed")}
    if got != expect:
        raise IdentityMismatch(f"done rollout {d} does not match the plan (stored {got}, plan {expect})")


def _check_parent(unit: Path, cid: str, prompt) -> dict:
    from entropytrail.runner import IdentityMismatch, _check_done_unit

    if not is_done(unit):
        raise IdentityMismatch(f"parent unit {unit} is not done")
    meta = json.loads((unit / "meta.json").read_text(encoding="utf-8"))
    if meta.get("condition_id") != cid:
        raise IdentityMismatch(f"parent unit {unit} belongs to {meta.get('condition_id')}, not {cid}")
    _check_done_unit(unit, prompt)
    return meta


def run_rollouts(plan: dict, cond, ps, model, tokenizer, root: Path, deadline, *,
                 fingerprint: str, now=datetime.now, io=None, plan_name: str = "",
                 plan_sha256: str = "") -> dict:
    """Run all rollouts of a fork plan that are not done yet (resumable, deadline-aware).

    Same contract as ``runner.run_units``: a rollout interrupted by the deadline is
    discarded, finished rollouts are never recomputed, one runlog entry per call (also on
    error, which is re-raised). See docs/DATA.md for the rollout layout and meta keys.
    """
    import dataclasses
    import shutil

    from entropytrail.model import Interrupted, generate_recorded, provenance, tokenizer_io
    from entropytrail.runner import IdentityMismatch, _move_stale, _sha256, check_condition
    from entropytrail.storage import Recording, TokenRecord, clean_partials, run_dir, write_run

    root = Path(root)
    cid = cond.condition_id
    started = now()
    prov = provenance()
    done = skipped = aborted = 0
    stale_moved: list[str] = []
    error: str | None = None
    prompts = {p.id: p for p in ps.prompts}

    def expired() -> bool:
        return deadline is not None and now() >= deadline

    try:
        if plan.get("condition_id") != cid:
            raise IdentityMismatch(f"fork plan is for {plan.get('condition_id')}, run is {cid}")
        _system, system_sha = check_condition(root, cond, ps, fingerprint, prov, started)
        clean_partials(root / cid)
        eos_ids, token_texts = io if io is not None else tokenizer_io(tokenizer)
        stop = False
        for app in plan["apps"]:
            if stop:
                break
            pid, sample = app["prompt_id"], int(app["sample"])
            prompt = prompts[pid]
            unit = run_dir(root, cid, pid, sample)
            pmeta = _check_parent(unit, cid, prompt)
            prompt_ids = [int(i) for i in pmeta["prompt_ids"]]
            parent_ids = [int(i) for i in read_tokens(unit).column("token_id").to_pylist()]
            # the parent must still be the app the plan was built on
            if app.get("n_tokens") is not None and len(parent_ids) != int(app["n_tokens"]):
                raise IdentityMismatch(f"parent {unit} has {len(parent_ids)} tokens, plan says {app['n_tokens']}")
            for site in app["sites"]:
                p_ = int(site["pos"])
                if p_ >= len(parent_ids) or parent_ids[p_] != int(site["actual_id"]):
                    raise IdentityMismatch(f"parent {unit} token at {p_} differs from the plan's actual_id")
            for site in app["sites"]:
                if stop:
                    break
                pos = int(site["pos"])
                for variant in VARIANTS:
                    if variant not in site["variants"]:
                        continue
                    forced = int(site["variants"][variant])
                    seed = _rollout_seed(cid, pid, sample, pos, variant)
                    d = rollout_dir(unit, pos, variant)
                    expect = {"parent": f"{pid}/{sample:03d}", "pos": pos, "variant": variant,
                              "forced_token_id": forced, "seed": seed}
                    if is_done(d):
                        _check_done_rollout(d, expect)
                        skipped += 1
                        continue
                    if expired():
                        stop = True
                        break
                    if d.exists():
                        stale_moved.append(str(_move_stale(d, now()).relative_to(root)))
                    t_start = now()
                    budget = cond.max_tokens - pos - 1
                    if forced in eos_ids or budget <= 0:
                        cont, stats, seconds = [], [], 0.0
                        finish = "eos" if forced in eos_ids else "max_tokens"
                    else:
                        try:
                            rec = generate_recorded(
                                model, prompt_ids + parent_ids[:pos] + [forced], seed=seed,
                                temperature=cond.temperature, max_tokens=budget,
                                eos_ids=eos_ids, token_texts=token_texts, should_stop=expired)
                        except Interrupted:
                            shutil.rmtree(d.parent / (d.name + ".partial"), ignore_errors=True)
                            aborted += 1
                            stop = True
                            break
                        cont = [t.token_id for t in rec.tokens]
                        stats = [t.stats for t in rec.tokens]
                        finish, seconds = rec.finish_reason, rec.seconds
                    ends_eos = finish == "eos" and bool(cont) and cont[-1] in eos_ids
                    body = cont[:-1] if ends_eos else cont
                    # decode the whole sequence: detokenizers merge bytes across token boundaries
                    if forced in eos_ids:  # EOS contributes no text, as in the grid apps
                        full = list(token_texts(parent_ids[:pos])) + [""]
                    else:
                        full = token_texts(parent_ids[:pos] + [forced] + body)
                    prefix_text = "".join(full[:pos + 1])
                    cont_texts = list(full[pos + 1:]) + ([""] if ends_eos else [])
                    records, off = [], 0
                    for k, (tid, txt, st) in enumerate(zip(cont, cont_texts, stats)):
                        records.append(TokenRecord(pos=k, token_id=tid, text=txt, char_offset=off, stats=st))
                        off += len(txt)
                    text = prefix_text + "".join(cont_texts)
                    assert text[len(prefix_text):] == "".join(r.text for r in records)
                    meta = dict(prov)
                    meta.update(
                        condition=dataclasses.asdict(cond), condition_id=cid,
                        prompt=dataclasses.asdict(prompt), prompt_set_version=ps.version,
                        system_prompt_version=cond.system_prompt_version,
                        system_prompt_sha256=system_sha, prompt_text_sha256=_sha256(prompt.text),
                        prompt_ids=prompt_ids, sample=sample, seed=seed,
                        weights_fingerprint=fingerprint,
                        started_at=t_start.isoformat(), finished_at=now().isoformat(),
                        fork={
                            "parent": f"{pid}/{sample:03d}", "pos": pos, "kind": site["kind"],
                            "variant": variant, "forced_token_id": forced,
                            "actual_token_id": int(parent_ids[pos]), "H": float(site["H"]),
                            "prefix_len": len(prompt_ids) + pos + 1,
                            "continuation_char_offset": len(prefix_text),
                            "plan": plan_name, "plan_sha256": plan_sha256,
                        },
                    )
                    write_run(d, Recording(prompt_ids=prompt_ids + parent_ids[:pos] + [forced],
                                           tokens=records, text=text, finish_reason=finish,
                                           seconds=seconds), meta)
                    done += 1
    except BaseException as e:
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        entry = {
            "started": started.isoformat(), "ended": now().isoformat(), "kind": "forks",
            "plan": plan_name, "condition_id": cid, "done": done, "skipped": skipped,
            "aborted": aborted, "error": error, "git_commit": prov["git_commit"],
            "weights_fingerprint": fingerprint, "stale_moved": stale_moved,
        }
        root.mkdir(parents=True, exist_ok=True)
        with open(root / "runlog.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    return entry
