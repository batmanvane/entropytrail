"""Morning job after a fork night (WP5): evaluate new rollouts, update the importance table,
add a journal entry and commit those two files (no push).

    uv run --frozen --offline --group eval --group analysis python scripts/morning_forks.py

Started by the LaunchAgent de.entropytrail.morning at 06:30 (ops/README.md). Does nothing if
ops/night.env has no ET_FORK_PLAN, if a night run is still active, or if the plan was already
reported complete. Never commits files other than the importance CSV and docs/JOURNAL.md,
and skips the commit if either had uncommitted changes before the job started.
ET_MORNING_DRY=1 runs eval and collect but only prints the journal entry (no write, no commit).
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COND = Path("/Volumes/Data/entropytrail/qwen35-9b-bf16-mlx-1195d88b")
RUNLOG = Path("/Volumes/Data/entropytrail/runlog.jsonl")


def log(msg: str) -> None:
    print(f"{datetime.now().isoformat(timespec='seconds')} morning: {msg}", flush=True)


def fork_plan() -> Path | None:
    m = re.search(r'^ET_FORK_PLAN="\$\{ET_FORK_PLAN-([^}]*)\}"', (ROOT / "ops" / "night.env").read_text(), re.M)
    return ROOT / m.group(1) if m and m.group(1) else None


def progress(plan: dict, cond: Path, since: datetime) -> dict:
    total = done = done_since = 0
    tps = []
    for app in plan["apps"]:
        unit = Path(cond) / app["prompt_id"] / f"{int(app['sample']):03d}" / "forks"
        for s in app["sites"]:
            for v in s["variants"]:
                total += 1
                meta = unit / f"{int(s['pos']):05d}" / v / "meta.json"
                try:
                    m = json.loads(meta.read_text())
                except (OSError, ValueError):
                    continue
                if m.get("status") != "done":
                    continue
                done += 1
                if datetime.fromisoformat(m["finished_at"]) >= since:
                    done_since += 1
                    tps.append(float(m.get("tokens_per_s") or 0))
    return {"total": total, "done": done, "done_since": done_since,
            "mean_tok_s_since": round(sum(tps) / len(tps), 1) if tps else None}


def fork_runs_since(runlog: Path, since: datetime) -> list[dict]:
    out = []
    for line in Path(runlog).read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("kind") == "forks" and datetime.fromisoformat(e["started"]) >= since:
            out.append(e)
    return out


def dirty_paths(repo: Path, paths: list[str]) -> list[str]:
    r = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", *paths],
                       capture_output=True, text=True, check=True)
    return [ln[3:] for ln in r.stdout.splitlines() if ln.strip()]


def _run(*cmd: str) -> int:
    log("run " + " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT).returncode


def main() -> int:
    plan_path = fork_plan()
    if plan_path is None:
        log("no ET_FORK_PLAN in ops/night.env, nothing to do")
        return 0
    if subprocess.run(["pgrep", "-f", "et run-window --condition"], capture_output=True).returncode == 0:
        log("a night run is still active, skipping")
        return 0
    name = plan_path.stem.removeprefix("plan-")
    csv_rel = f"results/forks/importance-{name}.csv"
    journal = ROOT / "docs" / "JOURNAL.md"
    marker = f"fork {name}: complete"
    if marker in journal.read_text():
        log("plan already reported complete, nothing to do")
        return 0
    foreign = dirty_paths(ROOT, [csv_rel, "docs/JOURNAL.md"])
    plan = json.loads(plan_path.read_text())
    prompts = sorted({a["prompt_id"] for a in plan["apps"]})

    rc_eval = _run(sys.executable, "-m", "entropytrail.cli", "eval", "--root", str(COND),
                   "--prompt-ids", *prompts, "--forks")
    rc_coll = _run(sys.executable, "scripts/collect_forks.py", "--plan", str(plan_path.relative_to(ROOT)))

    now = datetime.now()
    since = (now - timedelta(days=1)).replace(hour=22, minute=0, second=0, microsecond=0)
    p = progress(plan, COND, since)
    runs = fork_runs_since(RUNLOG, since)
    n_sites = sum(1 for _ in open(ROOT / csv_rel)) - 1 if (ROOT / csv_rel).exists() else 0
    errors = [e["error"] for e in runs if e.get("error")]
    complete = p["done"] == p["total"]
    lines = [
        f"\n## {now:%Y-%m-%d} (fork {name}, night {since:%Y-%m-%d}; automatic morning entry)\n",
        f"Night runs: {len(runs)} (done {sum(e.get('done', 0) for e in runs)}, "
        f"aborted {sum(e.get('aborted', 0) for e in runs)}, errors {len(errors)}"
        + (f": {errors[0]}" if errors else "") + "). "
        f"Rollouts finished since 22:00: {p['done_since']} (mean {p['mean_tok_s_since']} tok/s); "
        f"plan total {p['done']}/{p['total']}. `et eval --forks` exit {rc_eval}, collect exit {rc_coll}; "
        f"`{csv_rel}`: {n_sites} sites with importance.",
    ]
    if complete:
        lines.append(f" **{marker}** — next: H3 run (`scripts/analyze.py`) and the D2 scale decision.")
    if os.environ.get("ET_MORNING_DRY") == "1":
        log("dry run, journal entry would be:" + "".join(lines))
        return 0
    with open(journal, "a", encoding="utf-8") as fh:
        fh.write("".join(lines) + "\n")

    if foreign:
        log(f"not committing: {foreign} had uncommitted changes before this job")
        return 1
    paths = ["docs/JOURNAL.md"] + ([csv_rel] if (ROOT / csv_rel).exists() else [])
    subprocess.run(["git", "add", "--", *paths], cwd=ROOT, check=True)
    msg = f"results: fork {name} after night {since:%Y-%m-%d} ({p['done']}/{p['total']} rollouts, automatic)"
    rc = subprocess.run(["git", "commit", "-q", "-m", msg, "--", *paths], cwd=ROOT).returncode
    log(f"commit exit {rc}: {msg}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
