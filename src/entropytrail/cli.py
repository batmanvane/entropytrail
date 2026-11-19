"""The ``et`` command line interface."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


def next_occurrence(hhmm: str, now: datetime) -> datetime:
    """Next local occurrence of HH:MM strictly after ``now``."""
    h, m = (int(x) for x in hhmm.split(":"))
    t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return t if t > now else t + timedelta(days=1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="et", description="entropytrail recorder")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, prompts=True):
        sp.add_argument("--condition", required=True, type=Path)
        if prompts:
            sp.add_argument("--prompts", required=True, type=Path)

    g = sub.add_parser("generate", help="generate one unit, no deadline")
    common(g)
    g.add_argument("--prompt-id", required=True)
    g.add_argument("--sample", required=True, type=int)
    g.add_argument("--allow-dirty", action="store_true",
                   help="run even if the git working tree has uncommitted changes")

    r = sub.add_parser("run-window", help="run units until a deadline")
    common(r)
    r.add_argument("--samples", required=True, type=int)
    r.add_argument("--prompt-ids", nargs="+", default=None)
    r.add_argument("--until", required=True, metavar="HH:MM")
    r.add_argument("--allow-dirty", action="store_true",
                   help="run even if the git working tree has uncommitted changes")

    v = sub.add_parser("validate", help="teacher-forcing validation of the recorder")
    common(v)
    v.add_argument("--tokens", type=int, default=1500)
    v.add_argument("--chunk", type=int, default=512, help="teacher-forcing chunk size")

    b = sub.add_parser("bench", help="single-sequence throughput")
    common(b, prompts=False)
    b.add_argument("--tokens", type=int, default=1000)
    return p


def _setup_env() -> None:
    os.environ.setdefault("HF_HOME", "/Volumes/Models/huggingface")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _load(args, need_prompts=True):
    from entropytrail.condition import load_condition, load_prompts
    from entropytrail.model import load_model, weights_fingerprint

    cond = load_condition(args.condition)
    ps = load_prompts(args.prompts) if need_prompts else None
    model, tok = load_model(cond)
    return cond, ps, model, tok, weights_fingerprint(cond.model_repo)


def _run(args, units_fn, deadline) -> int:
    from entropytrail.model import provenance
    from entropytrail.runner import run_units
    from entropytrail.storage import data_root

    if provenance()["git_dirty"] and not args.allow_dirty:
        print("refusing to run: the git working tree has uncommitted changes, so meta.json "
              "could not name the code that produced the data; commit first or pass "
              "--allow-dirty", file=sys.stderr)
        return 2
    cond, ps, model, tok, fp = _load(args)
    res = run_units(units_fn(ps), cond, ps, model, tok, data_root(), deadline, fingerprint=fp)
    print(json.dumps(res))
    return 0


def cmd_generate(args) -> int:
    from entropytrail.runner import plan_units

    return _run(args, lambda ps: [
        u for u in plan_units(ps, args.sample + 1, [args.prompt_id]) if u.sample == args.sample
    ], None)


def cmd_run_window(args) -> int:
    from entropytrail.runner import plan_units

    deadline = next_occurrence(args.until, datetime.now())
    return _run(args, lambda ps: plan_units(ps, args.samples, args.prompt_ids), deadline)


def cmd_validate(args) -> int:
    from entropytrail.condition import derive_seed
    from entropytrail.model import (
        build_prompt_ids, generate_recorded, provenance, tokenizer_io,
    )
    from entropytrail.storage import data_root
    from entropytrail.teacher_forcing import validate_recording

    cond, ps, model, tok, fp = _load(args)
    prompt = ps.prompts[0]
    ids = build_prompt_ids(tok, ps.system_prompts[cond.system_prompt_version],
                           prompt.text, cond.thinking)
    seed = derive_seed(cond.condition_id, prompt.id, 0)
    eos_ids, texts = tokenizer_io(tok)
    kw = dict(seed=seed, temperature=cond.temperature, max_tokens=args.tokens,
              eos_ids=eos_ids, token_texts=texts)
    rec = generate_recorded(model, ids, **kw)
    t0 = time.monotonic()
    res = validate_recording(model, rec, chunk=args.chunk)
    score_s = time.monotonic() - t0
    rec2 = generate_recorded(model, ids, **kw)
    res["same_tokens"] = [t.token_id for t in rec.tokens] == [t.token_id for t in rec2.tokens]
    n = len(rec.tokens)
    res.update(
        condition_id=cond.condition_id, prompt_id=prompt.id, seed=seed, n_tokens=n,
        generate_tokens_per_s=(n / rec.seconds) if rec.seconds else 0.0,
        # validate_recording scores the sequence twice (chunk=1 and chunk=N)
        score_tokens_per_s=(2 * n / score_s) if score_s else 0.0,
        weights_fingerprint=fp,
    )
    res.update(provenance())
    out = data_root() / "_validation" / cond.condition_id / _stamp()
    out.mkdir(parents=True)
    (out / "validation.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0 if res["ok"] and res["same_tokens"] else 1


def cmd_bench(args) -> int:
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    from entropytrail.model import build_prompt_ids
    from entropytrail.storage import data_root

    cond, ps, model, tok, _ = _load(_BenchArgs(args), need_prompts=False)
    user = "Write a long, detailed single-file HTML5 canvas game with many features."
    ids = build_prompt_ids(tok, "You are a helpful assistant.", user, cond.thinking)
    mx.random.seed(0)
    t0 = time.monotonic()
    t_first = None
    n = 0
    for _tok, _lp in generate_step(mx.array(ids), model, max_tokens=args.tokens,
                                   sampler=make_sampler(temp=cond.temperature)):
        n += 1
        if t_first is None:
            t_first = time.monotonic()
    t_end = time.monotonic()
    prefill_s = (t_first or t_end) - t0
    decode_s = t_end - (t_first or t_end)
    res = {
        "condition_id": cond.condition_id, "prompt_tokens": len(ids), "tokens": n,
        "prefill_s": prefill_s, "prefill_tokens_per_s": len(ids) / prefill_s if prefill_s else 0.0,
        "decode_s": decode_s,
        "decode_tokens_per_s": (n - 1) / decode_s if decode_s and n > 1 else 0.0,
    }
    out = data_root() / "_bench"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{_stamp()}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0


class _BenchArgs:
    def __init__(self, a):
        self.condition = a.condition


def main(argv: list[str] | None = None) -> None:
    _setup_env()
    args = build_parser().parse_args(argv)
    fn = {"generate": cmd_generate, "run-window": cmd_run_window,
          "validate": cmd_validate, "bench": cmd_bench}[args.cmd]
    sys.exit(fn(args))
