import importlib.util
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("morning_forks", ROOT / "scripts" / "morning_forks.py")
mf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mf)

PLAN = {"apps": [{"prompt_id": "p1", "sample": 0, "sites": [
    {"pos": 5, "variants": {"alt1": 1, "same1": 2}}, {"pos": 9, "variants": {"alt1": 3}}]}]}


def _done(d, finished, tps=10.0):
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"status": "done", "finished_at": finished, "tokens_per_s": tps}))


def test_progress_counts_done_rollouts(tmp_path):
    u = tmp_path / "p1" / "000" / "forks"
    _done(u / "00005" / "alt1", "2026-11-23T01:00:00")
    _done(u / "00009" / "alt1", "2026-11-22T10:55:00", tps=20.0)
    (u / "00005" / "same1.partial").mkdir(parents=True)
    p = mf.progress(PLAN, tmp_path, since=datetime(2026, 11, 22, 22, 0))
    assert p == {"total": 3, "done": 2, "done_since": 1, "mean_tok_s_since": 10.0}


def test_runlog_entries_since(tmp_path):
    log = tmp_path / "runlog.jsonl"
    log.write_text("\n".join(json.dumps(e) for e in [
        {"started": "2026-11-21T23:00:01", "kind": "forks", "done": 1},
        {"started": "2026-11-22T23:00:01", "kind": "forks", "done": 40, "aborted": 1, "error": None},
        {"started": "2026-11-22T23:00:02", "done": 3},
    ]) + "\n")
    e = mf.fork_runs_since(log, datetime(2026, 11, 22, 22, 0))
    assert [x["done"] for x in e] == [40]


def test_paths_dirty_detects_foreign_changes(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    f = tmp_path / "a.txt"
    f.write_text("x")
    subprocess.run(["git", "-C", str(tmp_path), "add", "a.txt"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "i"],
                   check=True)
    assert mf.dirty_paths(tmp_path, ["a.txt"]) == []
    f.write_text("y")
    assert mf.dirty_paths(tmp_path, ["a.txt"]) == ["a.txt"]
