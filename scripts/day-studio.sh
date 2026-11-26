#!/bin/bash
# Day run window on the Studio for fork plans (WP5) or a grid queue (WP12, ET_DAY_GRID=1): 07:15-22:45, started by the user
# LaunchAgent de.entropytrail.day. Ollama is left running (idle) as on earlier day runs
# (D-018). Runs only while ops/night.env sets ET_FORK_PLAN, never while another
# `et run-window` is active, and kills its own run at 22:50 at the latest so it can never
# overlap the 23:00 night run (two 9B models do not fit into 32 GB).
# ET_DRY_RUN=1 echoes the run command; ET_HANG_RUN=1 (dry run only) simulates a hang.
set -euo pipefail

REPO="${ET_REPO:-/Users/robert/projects/entropytrail}"
MODELS_DIR="${ET_MODELS_DIR:-/Volumes/Models}"
DATA_DIR="${ET_DATA_DIR:-/Volumes/Data}"

log() { echo "$(date '+%Y-%m-%dT%H:%M:%S%z') $*"; }

log "day start"
hour=$((10#${ET_NOW_HOUR:-$(date +%H)}))
if [ "$hour" -lt 7 ] || [ "$hour" -ge 22 ]; then
  log "hour $hour is outside the 07-22 day window, skipping"
  exit 0
fi

# shellcheck disable=SC1091
source "$REPO/ops/night.env"
if [ -n "${ET_FORK_PLAN:-}" ]; then
  queue=(--fork-plan "$ET_FORK_PLAN")
  if [ "${ET_DAY_GRID:-0}" = "1" ]; then
    log "warning: ET_FORK_PLAN and ET_DAY_GRID are both set; running the fork plan"
  fi
elif [ "${ET_DAY_GRID:-0}" = "1" ]; then
  queue=(--samples "${ET_SAMPLES:-20}")
  if [ -n "${ET_PROMPT_IDS:-}" ]; then
    # shellcheck disable=SC2206
    queue+=(--prompt-ids ${ET_PROMPT_IDS})
  fi
else
  log "neither ET_FORK_PLAN nor ET_DAY_GRID=1 in ops/night.env, nothing to do"
  exit 0
fi
for d in "$MODELS_DIR" "$DATA_DIR"; do
  if [ ! -d "$d" ]; then log "required volume $d not mounted, skipping"; exit 0; fi
done
active="${ET_OTHER_RUN_ACTIVE:-}"
if [ -z "$active" ]; then
  if pgrep -f "et run-window --condition" >/dev/null; then active=1; else active=0; fi
fi
if [ "$active" = "1" ]; then
  log "another et run-window is already active, skipping"
  exit 0
fi

run_step() {
  if [ "${ET_DRY_RUN:-0}" = "1" ]; then
    if [ "${ET_HANG_RUN:-0}" = "1" ]; then
      echo "DRY: simulated hanging run-window"
      exec sleep 600 </dev/null >/dev/null 2>&1
    fi
    echo "DRY: uv --directory $REPO run --frozen --offline et run-window --condition $ET_CONDITION" \
      "--prompts $ET_PROMPTS ${queue[*]} --until 22:45"
    return 0
  fi
  HF_HOME=/Volumes/Models/huggingface "${ET_UV:-/opt/homebrew/bin/uv}" --directory "$REPO" run --frozen --offline \
    et run-window --condition "$ET_CONDITION" --prompts "$ET_PROMPTS" \
    "${queue[@]}" --until 22:45
}

if [ -n "${ET_WATCHDOG_SECONDS:-}" ]; then
  limit="$ET_WATCHDOG_SECONDS"
else
  limit=$(( $(date -j -v22H -v50M -v0S +%s) - $(date +%s) ))
fi
grace="${ET_WATCHDOG_GRACE:-30}"

# signal a process and all its descendants (uv -> python), leaves first
kill_tree() {
  local sig=$1 pid=$2 child
  for child in $(pgrep -P "$pid" 2>/dev/null); do kill_tree "$sig" "$child"; done
  kill "-$sig" "$pid" 2>/dev/null || true
}

run_step &
run_pid=$!
(
  sp=""
  trap '[ -n "$sp" ] && kill "$sp" 2>/dev/null; exit 0' TERM
  sleep "$limit" </dev/null >/dev/null 2>&1 & sp=$!; wait "$sp"
  log "watchdog: day run still active at 22:50, terminating it"
  # kill only this run's process tree, never a pattern (the night run must survive)
  kill_tree TERM "$run_pid"
  sleep "$grace" </dev/null >/dev/null 2>&1 & sp=$!; wait "$sp"
  kill_tree KILL "$run_pid"
) &
watchdog_pid=$!

run_rc=0
wait "$run_pid" || run_rc=$?
kill "$watchdog_pid" 2>/dev/null || true
wait "$watchdog_pid" 2>/dev/null || true
log "day end (exit code $run_rc)"
exit "$run_rc"
