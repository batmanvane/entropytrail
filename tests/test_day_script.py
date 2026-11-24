import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "day-studio.sh"


def _run(tmp_path, **extra):
    env = {**os.environ, "ET_DRY_RUN": "1", "ET_REPO": str(ROOT), "ET_MODELS_DIR": str(tmp_path),
           "ET_DATA_DIR": str(tmp_path), "ET_NOW_HOUR": "10", "ET_FORK_PLAN": "results/forks/plan-pilot.json",
           "ET_PROMPT_IDS": "", "ET_OTHER_RUN_ACTIVE": "0", "ET_DAY_GRID": "0", **extra}
    return subprocess.run(["/bin/bash", str(SCRIPT)], capture_output=True, text=True, env=env, cwd=tmp_path)


def test_day_run_command(tmp_path):
    r = _run(tmp_path)
    assert r.returncode == 0, r.stderr
    line = [ln for ln in r.stdout.splitlines() if "run-window" in ln][0]
    assert "--fork-plan results/forks/plan-pilot.json" in line and "--until 22:45" in line
    assert "run --frozen --offline et run-window" in line
    assert "launchctl" not in r.stdout  # Ollama is left alone during the day


def test_outside_day_window_skips(tmp_path):
    for h in ("06", "23"):
        r = _run(tmp_path, ET_NOW_HOUR=h)
        assert r.returncode == 0 and "run-window" not in r.stdout and "outside" in r.stdout


def test_no_fork_plan_skips(tmp_path):
    r = _run(tmp_path, ET_FORK_PLAN="")
    assert r.returncode == 0 and "run-window" not in r.stdout


def test_other_run_active_skips(tmp_path):
    r = _run(tmp_path, ET_OTHER_RUN_ACTIVE="1")
    assert r.returncode == 0 and "--until" not in r.stdout and "already active" in r.stdout


def test_watchdog_kills_own_run_only(tmp_path):
    r = _run(tmp_path, ET_HANG_RUN="1", ET_WATCHDOG_SECONDS="1", ET_WATCHDOG_GRACE="1")
    assert "watchdog" in r.stdout and "pkill" not in r.stdout


def test_watchdog_kills_grandchildren(tmp_path):
    fake = tmp_path / "repo"
    (fake / "ops").mkdir(parents=True)
    (fake / "ops" / "night.env").write_text('ET_CONDITION=c\nET_PROMPTS=p\nET_FORK_PLAN="${ET_FORK_PLAN-x.json}"\n')
    bin_ = tmp_path / "uv"
    marker = tmp_path / "grandchild.pid"
    bin_.write_text(f"#!/bin/bash\n/bin/bash -c 'echo $$ > {marker}; sleep 600' &\nwait\n")
    bin_.chmod(0o755)
    env = {**os.environ, "ET_REPO": str(fake), "ET_MODELS_DIR": str(tmp_path), "ET_DATA_DIR": str(tmp_path),
           "ET_NOW_HOUR": "10", "ET_OTHER_RUN_ACTIVE": "0", "ET_UV": str(bin_),
           "ET_WATCHDOG_SECONDS": "2", "ET_WATCHDOG_GRACE": "1"}
    r = subprocess.run(["/bin/bash", str(SCRIPT)], capture_output=True, text=True, env=env, timeout=30)
    assert "watchdog" in r.stdout
    pid = int(marker.read_text())
    assert subprocess.run(["kill", "-0", str(pid)], capture_output=True).returncode != 0


def test_day_grid_queue(tmp_path):
    r = _run(tmp_path, ET_FORK_PLAN="", ET_DAY_GRID="1", ET_SAMPLES="20")
    line = [ln for ln in r.stdout.splitlines() if "run-window" in ln][0]
    assert "--samples 20" in line and "--until 22:45" in line and "--fork-plan" not in line


def test_day_grid_with_prompt_ids(tmp_path):
    r = _run(tmp_path, ET_FORK_PLAN="", ET_DAY_GRID="1", ET_SAMPLES="20", ET_PROMPT_IDS="free-01 free-02")
    line = [ln for ln in r.stdout.splitlines() if "run-window" in ln][0]
    assert "--prompt-ids free-01 free-02" in line


def test_fork_plan_wins_over_day_grid_with_warning(tmp_path):
    r = _run(tmp_path, ET_DAY_GRID="1")
    line = [ln for ln in r.stdout.splitlines() if "run-window" in ln][0]
    assert "--fork-plan" in line and "--samples" not in line and "warning" in r.stdout


def test_neither_queue_does_nothing(tmp_path):
    r = _run(tmp_path, ET_FORK_PLAN="", ET_DAY_GRID="0")
    assert r.returncode == 0 and "run-window" not in r.stdout
