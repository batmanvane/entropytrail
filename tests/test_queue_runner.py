import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("queue_runner", ROOT / "scripts" / "queue_runner.py")
qr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qr)

QUEUE = """
# comment
grid configs/conditions/t0p7.yaml 20
eval /data/t0p7
forks configs/conditions/t10.yaml results/forks/plan-main.json
eval-forks /data/t10 game-tight-01
rescore configs/conditions/t10.yaml
"""


def test_parse_queue():
    jobs = qr.parse_queue(QUEUE)
    assert [j.kind for j in jobs] == ["grid", "eval", "forks", "eval-forks", "rescore"]
    assert jobs[0].args == ["configs/conditions/t0p7.yaml", "20"]
    with pytest.raises(ValueError):
        qr.parse_queue("bake bread now")
    with pytest.raises(ValueError):
        qr.parse_queue("rescore a.yaml b.yaml")


def test_commands():
    jobs = qr.parse_queue(QUEUE)
    py = "/venv/bin/python"
    et = "/venv/bin/et"  # the `et` entry point, so `pgrep -f "et run-window --condition"` sees the run
    assert qr.command(jobs[0], py) == [et, "run-window", "--condition",
                                       "configs/conditions/t0p7.yaml", "--prompts", "prompts/prompts.yaml",
                                       "--samples", "20"]
    assert qr.command(jobs[1], py) == [et, "eval", "--root", "/data/t0p7"]
    assert "--fork-plan" in qr.command(jobs[2], py) and "--until" not in qr.command(jobs[2], py)
    assert qr.command(jobs[3], py)[-4:] == ["/data/t10", "--forks", "--prompt-ids", "game-tight-01"]
    assert qr.command(jobs[4], py) == [et, "rescore", "--condition", "configs/conditions/t10.yaml",
                                       "--prompts", "prompts/prompts.yaml"]
    assert qr.is_complete(jobs[4], 0, json.dumps({"done": 3, "skipped": 0, "failed": 0}))


def test_complete_detection():
    j_run, j_eval = qr.parse_queue("grid c.yaml 20\neval /d")
    ok = json.dumps({"done": 0, "skipped": 420, "aborted": 0, "error": None})
    assert qr.is_complete(j_run, 0, "log\n" + ok)
    assert not qr.is_complete(j_run, 0, json.dumps({"done": 3, "aborted": 0, "error": None}))
    assert not qr.is_complete(j_run, 2, "")  # e.g. dirty tree
    assert qr.is_complete(j_eval, 0, json.dumps({"done": 5, "skipped": 0, "failed": 0}))
    assert not qr.is_complete(j_eval, 1, json.dumps({"done": 5, "skipped": 0, "failed": 2}))


def test_run_job_kills_a_stalled_process(tmp_path):
    cmd = [sys.executable, "-c", "import time; time.sleep(60)"]
    rc, out, stalled = qr.run_job(cmd, progress=lambda: 0, stall_s=1.0, poll_s=0.2, cwd=tmp_path)
    assert stalled and rc != 0


def test_run_job_returns_output(tmp_path):
    cmd = [sys.executable, "-c", "print('a'); print('{\"done\": 0}')"]
    rc, out, stalled = qr.run_job(cmd, progress=lambda: 0, stall_s=30, poll_s=0.2, cwd=tmp_path)
    assert rc == 0 and not stalled and out.strip().endswith('{"done": 0}')


def test_queue_runs_jobs_in_order_until_complete(tmp_path):
    q = tmp_path / "queue.txt"
    q.write_text("grid c.yaml 20\neval /d\n")
    calls = []
    results = iter([(0, json.dumps({"done": 4, "aborted": 0, "error": None}), False),
                    (0, json.dumps({"done": 0, "aborted": 0, "error": None}), False),
                    (0, json.dumps({"done": 0, "skipped": 4, "failed": 0}), False)])

    def run(cmd, **kw):
        calls.append(cmd[1])
        return next(results)

    qr.run_queue(q, python="py", run=run, other_active=lambda: False, sleep=lambda s: None,
                 log=lambda m: None, progress_for=lambda job: (lambda: 0))
    assert calls == ["run-window", "run-window", "eval"]


def test_queue_waits_for_other_run_and_retries_after_failure(tmp_path):
    q = tmp_path / "queue.txt"
    q.write_text("grid c.yaml 20\n")
    active = iter([True, True, False, False])
    slept = []
    results = iter([(2, "refusing: dirty", False), (0, json.dumps({"done": 0, "aborted": 0, "error": None}), False)])
    qr.run_queue(q, python="py", run=lambda cmd, **kw: next(results), other_active=lambda: next(active),
                 sleep=slept.append, log=lambda m: None, progress_for=lambda job: (lambda: 0))
    assert len(slept) == 3  # two waits for the other run, one back-off after the failed attempt
