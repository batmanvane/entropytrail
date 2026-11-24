"""24/7 job queue on the Studio (D-039): runs the jobs of ops/queue.txt in order, without windows.

    uv run --frozen --offline --group eval --group analysis python scripts/queue_runner.py

Started by the user LaunchAgent de.entropytrail.queue (KeepAlive). One line per job:

    grid <condition.yaml> <samples>              et run-window --samples (no deadline)
    forks <condition.yaml> <plan.json>           et run-window --fork-plan (no deadline)
    eval <condition dir> [prompt ids…]           et eval
    eval-forks <condition dir> [prompt ids…]     et eval --forks

A job is repeated until it reports nothing left (run-window: done 0, aborted 0, no error;
eval: exit 0), then the next job starts; finished units are never recomputed, so a restart
only re-checks. A job is never started while another `et run-window` runs. If a job makes no
progress (no new finished unit) for STALL_S seconds its process group is killed and the job
restarts (e.g. a macOS privacy prompt blocking open()). A refused or failed attempt (dirty
git tree, crash) is retried after BACKOFF_S. ops/queue.txt is re-read before every job, so
jobs can be appended while the queue runs. Exit 0 when every job is complete.
"""

import glob
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/Volumes/Data/entropytrail")
PROMPTS = "prompts/prompts.yaml"
STALL_S = 2 * 3600
POLL_S = 300
BACKOFF_S = 600
KINDS = {"grid": 2, "forks": 2, "eval": 1, "eval-forks": 1}


@dataclass
class Job:
    kind: str
    args: list[str]

    def __str__(self) -> str:
        return " ".join([self.kind, *self.args])


def parse_queue(text: str) -> list[Job]:
    jobs = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        kind, *args = line.split()
        if kind not in KINDS or len(args) < KINDS[kind] or (kind in ("grid", "forks") and len(args) != 2):
            raise ValueError(f"bad queue line: {raw!r}")
        jobs.append(Job(kind, args))
    return jobs


def command(job: Job, python: str) -> list[str]:
    # the `et` entry point (not `python -m`), so `pgrep -f "et run-window --condition"` in the
    # night/day scripts and in other_run_active() recognises a queue run
    base = [str(Path(python).parent / "et")]
    if job.kind in ("grid", "forks"):
        cond, what = job.args
        flag = ["--samples", what] if job.kind == "grid" else ["--fork-plan", what]
        return base + ["run-window", "--condition", cond, "--prompts", PROMPTS, *flag]
    cmd = base + ["eval", "--root", job.args[0]]
    if job.kind == "eval-forks":
        cmd.append("--forks")
    if len(job.args) > 1:
        cmd += ["--prompt-ids", *job.args[1:]]
    return cmd


def _last_json(out: str) -> dict | None:
    for line in reversed(out.strip().splitlines()):
        try:
            return json.loads(line)
        except ValueError:
            continue
    return None


def is_complete(job: Job, rc: int, out: str) -> bool:
    if rc != 0:
        return False
    if job.kind in ("eval", "eval-forks"):
        return True
    r = _last_json(out)
    return bool(r) and r.get("done") == 0 and r.get("aborted") == 0 and r.get("error") is None


def run_job(cmd: list[str], *, progress, stall_s: float, poll_s: float, cwd: Path) -> tuple[int, str, bool]:
    """Run ``cmd``; kill its process group if ``progress()`` does not grow for ``stall_s``."""
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)
    chunks: list[str] = []
    reader = threading.Thread(target=lambda: chunks.extend(proc.stdout or []), daemon=True)
    reader.start()
    last, last_change, stalled = progress(), time.monotonic(), False
    while proc.poll() is None:
        time.sleep(poll_s)
        now = progress()
        if now != last:
            last, last_change = now, time.monotonic()
        elif time.monotonic() - last_change > stall_s:
            stalled = True
            for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    proc.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
            break
    proc.wait()
    reader.join(timeout=5)
    return proc.returncode, "".join(chunks), stalled


def progress_for(job: Job):
    """Number of finished units of the job (meta.json or eval.json files)."""
    if job.kind in ("grid", "forks"):
        import yaml

        from entropytrail.condition import Condition

        cid = Condition(**yaml.safe_load((ROOT / job.args[0]).read_text())).condition_id
        pattern = "*/[0-9][0-9][0-9]/meta.json" if job.kind == "grid" else "*/[0-9][0-9][0-9]/forks/*/*/meta.json"
        base = DATA / cid
    else:
        base = Path(job.args[0])
        pattern = "*/[0-9][0-9][0-9]/eval/eval.json"
        if job.kind == "eval-forks":
            pattern = "*/[0-9][0-9][0-9]/**/eval/eval.json"
    return lambda: len(glob.glob(str(base / pattern), recursive=True))


def other_run_active() -> bool:
    return subprocess.run(["pgrep", "-f", "et run-window --condition"],
                          capture_output=True).returncode == 0


def log(msg: str) -> None:
    print(f"{datetime.now().isoformat(timespec='seconds')} queue: {msg}", flush=True)


def run_queue(queue: Path, *, python: str, run=run_job, other_active=other_run_active, sleep=time.sleep,
              log=log, progress_for=progress_for, stall_s: float = STALL_S, poll_s: float = POLL_S) -> None:
    i = 0
    while True:
        jobs = parse_queue(Path(queue).read_text())
        if i >= len(jobs):
            log("queue complete")
            return
        job = jobs[i]
        while other_active():
            log(f"another et run-window is active, waiting before: {job}")
            sleep(poll_s)
        log(f"start: {job}")
        rc, out, stalled = run(command(job, python), progress=progress_for(job), stall_s=stall_s,
                               poll_s=poll_s, cwd=ROOT)
        tail = out.strip().splitlines()[-1:] if out.strip() else []
        log(f"end: {job} (exit {rc}{', killed after stall' if stalled else ''}) {tail[0][:300] if tail else ''}")
        if is_complete(job, rc, out):
            log(f"complete: {job}")
            i += 1
        elif rc != 0 or stalled:
            sleep(BACKOFF_S)


def main() -> int:
    run_queue(ROOT / "ops" / "queue.txt", python=sys.executable)
    return 0


if __name__ == "__main__":
    sys.exit(main())
