import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "night-studio.sh"


def _run(tmp_path, **extra):
    env = {
        **os.environ,
        "ET_DRY_RUN": "1",
        "ET_REPO": str(ROOT),
        "ET_MODELS_DIR": str(tmp_path),
        "ET_DATA_DIR": str(tmp_path),
        "ET_NOW_HOUR": "23",
        **extra,
    }
    return subprocess.run(
        ["/bin/bash", str(SCRIPT)], capture_output=True, text=True, env=env, cwd=tmp_path
    )


def _cmd_lines(out):
    return [ln for ln in out.splitlines() if "launchctl" in ln or "run-window" in ln]


def test_dry_run_order(tmp_path):
    r = _run(tmp_path)
    assert r.returncode == 0, r.stderr
    lines = _cmd_lines(r.stdout)
    assert "bootout system/com.robert.ollama" in lines[0]
    assert "et run-window" in lines[1] and "--until 06:00" in lines[1]
    assert "--samples 20" in lines[1]
    assert "bootstrap system" in lines[-1] and "com.robert.ollama.plist" in lines[-1]
    assert len(lines) == 3


def test_failing_run_still_restarts_ollama_last(tmp_path):
    r = _run(tmp_path, ET_FAIL_RUN="1")
    assert r.returncode != 0
    lines = _cmd_lines(r.stdout)
    assert "bootstrap system" in lines[-1] and "com.robert.ollama.plist" in lines[-1]


def test_missing_mount_exits_but_restarts_ollama(tmp_path):
    r = _run(tmp_path, ET_MODELS_DIR=str(tmp_path / "nope"))
    assert r.returncode != 0
    assert "run-window" not in r.stdout
    assert "bootstrap system" in _cmd_lines(r.stdout)[-1]


def test_logs_start_and_end(tmp_path):
    r = _run(tmp_path)
    assert "night start" in r.stdout and "night end" in r.stdout


def test_outside_window_does_nothing(tmp_path):
    r = _run(tmp_path, ET_NOW_HOUR="08")
    assert r.returncode == 0
    assert "outside" in r.stdout
    assert "bootout" not in r.stdout and "run-window" not in r.stdout


def test_inside_window_hours_run(tmp_path):
    for h in ("23", "00", "02", "05"):
        r = _run(tmp_path, ET_NOW_HOUR=h)
        assert r.returncode == 0, h
        assert "bootout system/com.robert.ollama" in r.stdout, h
        assert "et run-window" in r.stdout, h
    r = _run(tmp_path, ET_NOW_HOUR="06")
    assert "run-window" not in r.stdout
