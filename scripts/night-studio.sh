#!/bin/bash
# Nightly run window on the Studio. Runs as root (LaunchDaemon) so it can
# stop/start the system Ollama daemon; `et` itself runs as user robert.
# ET_DRY_RUN=1 echoes every external command. ET_FAIL_RUN=1 (dry run only)
# makes the run step fail, to test that Ollama is still restarted.
# Ollama is restarted only if this run stopped it (not when skipping outside the
# window or aborting on an unmounted volume).
set -euo pipefail

REPO="${ET_REPO:-/Users/robert/projects/entropytrail}"
OLLAMA_LABEL="com.robert.ollama"
OLLAMA_PLIST="/Library/LaunchDaemons/${OLLAMA_LABEL}.plist"
MODELS_DIR="${ET_MODELS_DIR:-/Volumes/Models}"
DATA_DIR="${ET_DATA_DIR:-/Volumes/Data}"

log() { echo "$(date '+%Y-%m-%dT%H:%M:%S%z') $*"; }

# Run an external command, or just echo it in dry-run mode.
x() {
  if [ "${ET_DRY_RUN:-0}" = "1" ]; then echo "DRY: $*"; else "$@"; fi
}

# Set once this run has booted Ollama out; only then does the EXIT trap start it
# again (a skipped or aborted night never starts a deliberately stopped Ollama).
OLLAMA_BOOTED_OUT=0

on_exit() {
  local rc=$?
  trap - EXIT
  set +e
  if [ "$OLLAMA_BOOTED_OUT" = "1" ]; then
    x launchctl bootstrap system "$OLLAMA_PLIST" \
      || log "bootstrap returned non-zero (already loaded?), ignored"
  fi
  log "night end (exit code $rc)"
  exit "$rc"
}
trap on_exit EXIT

log "night start"

# Window guard first: launchd fires missed intervals after wake; only run 23:00-05:59.
hour=$((10#${ET_NOW_HOUR:-$(date +%H)}))
if [ "$hour" -lt 23 ] && [ "$hour" -ge 6 ]; then
  log "hour $hour is outside the 23-06 window, skipping"
  exit 0
fi

# shellcheck disable=SC1091
source "$REPO/ops/night.env"

if [ -n "${ET_FORK_PLAN:-}" ] && [ -n "${ET_PROMPT_IDS:-}" ]; then
  log "config error: ET_FORK_PLAN and ET_PROMPT_IDS are both set in ops/night.env, aborting"
  exit 2
fi

for d in "$MODELS_DIR" "$DATA_DIR"; do
  if [ ! -d "$d" ]; then
    log "required volume $d not mounted, aborting"
    exit 1
  fi
done

x launchctl bootout "system/$OLLAMA_LABEL" \
  || log "bootout returned non-zero (not loaded?), ignored"
OLLAMA_BOOTED_OUT=1

if [ "${ET_DRY_RUN:-0}" = "1" ] && [ "${ET_FAIL_RUN:-0}" = "1" ]; then
  echo "DRY: forced failure of run step"
  exit 1
fi

queue=(--samples "${ET_SAMPLES:-20}")
if [ -n "${ET_FORK_PLAN:-}" ]; then
  queue=(--fork-plan "$ET_FORK_PLAN")
elif [ -n "${ET_PROMPT_IDS:-}" ]; then
  # shellcheck disable=SC2206
  queue+=(--prompt-ids ${ET_PROMPT_IDS})
fi

run_step() {
  if [ "${ET_DRY_RUN:-0}" = "1" ] && [ "${ET_HANG_RUN:-0}" = "1" ]; then
    echo "DRY: simulated hanging run-window"
    exec sleep 600 </dev/null >/dev/null 2>&1
  fi
  x sudo -u robert -H env HF_HOME=/Volumes/Models/huggingface \
    /opt/homebrew/bin/uv --directory "$REPO" run --frozen --offline et run-window \
    --condition "$ET_CONDITION" --prompts "$ET_PROMPTS" \
    "${queue[@]}" --until 06:00
}

# Watchdog: `--until 06:00` is only checked between tokens. A run blocked elsewhere
# (e.g. a macOS privacy prompt on open(), 2026-11-19) would never reach it and Ollama
# would stay stopped. At 06:05 the watchdog terminates the run (TERM, then KILL after a
# grace period); the EXIT trap then restarts Ollama. An interrupted unit leaves only a
# .partial directory, which the next run cleans up.
if [ -n "${ET_WATCHDOG_SECONDS:-}" ]; then
  limit="$ET_WATCHDOG_SECONDS"
else
  now=$(date +%s)
  target=$(date -j -v6H -v5M -v0S +%s)
  if [ "$target" -le "$now" ]; then target=$((target + 86400)); fi
  limit=$((target - now))
  log "watchdog armed until $(date -j -r "$target" '+%Y-%m-%d 06:05')"
fi
grace="${ET_WATCHDOG_GRACE:-30}"

run_step &
run_pid=$!
# The sleeps run detached from stdout and are killed with the watchdog, so a finished
# run never leaves a sleeping child behind (it would hold the log pipe open).
(
  sp=""
  trap '[ -n "$sp" ] && kill "$sp" 2>/dev/null; exit 0' TERM
  sleep "$limit" </dev/null >/dev/null 2>&1 & sp=$!; wait "$sp"
  log "watchdog: run still active after ${limit}s, terminating"
  x pkill -TERM -f "et run-window --condition" || true
  kill -TERM "$run_pid" 2>/dev/null || true
  sleep "$grace" </dev/null >/dev/null 2>&1 & sp=$!; wait "$sp"
  x pkill -KILL -f "et run-window --condition" || true
  kill -KILL "$run_pid" 2>/dev/null || true
) &
watchdog_pid=$!

run_rc=0
wait "$run_pid" || run_rc=$?
kill "$watchdog_pid" 2>/dev/null || true
wait "$watchdog_pid" 2>/dev/null || true
exit "$run_rc"
