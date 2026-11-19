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

ids=()
if [ -n "${ET_PROMPT_IDS:-}" ]; then
  # shellcheck disable=SC2206
  ids=(--prompt-ids ${ET_PROMPT_IDS})
fi

x sudo -u robert -H env HF_HOME=/Volumes/Models/huggingface \
  /opt/homebrew/bin/uv --directory "$REPO" run --frozen --offline et run-window \
  --condition "$ET_CONDITION" --prompts "$ET_PROMPTS" \
  --samples "${ET_SAMPLES:-20}" ${ids[@]+"${ids[@]}"} --until 06:00
