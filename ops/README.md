# Night scheduling (Mac Studio)

A root LaunchDaemon `de.entropytrail.night` runs `scripts/night-studio.sh` daily at 23:00.
The script first checks the window (only 23:00-05:59; a missed 23:00 fired by launchd
after wake is skipped) and that `/Volumes/Models` and `/Volumes/Data` are mounted, then
stops the system Ollama daemon (`com.robert.ollama`), runs
`uv run --frozen --offline et run-window ... --until 06:00` as user robert, and restarts
Ollama on every exit path after it was stopped (EXIT trap), including failures. A skipped
night or an unmounted volume exits without touching Ollama.

`et run-window` refuses to run when the repository has uncommitted changes (the night run
does not pass `--allow-dirty`); the reason is in the night log, and no runlog entry is
written in that case. Commit before the window starts.

## Install

    sudo cp /Users/robert/projects/entropytrail/ops/launchd/de.entropytrail.night.plist /Library/LaunchDaemons/
    sudo chown root:wheel /Library/LaunchDaemons/de.entropytrail.night.plist
    sudo launchctl bootstrap system /Library/LaunchDaemons/de.entropytrail.night.plist

## Uninstall / pause the series

    sudo launchctl bootout system/de.entropytrail.night
    sudo rm /Library/LaunchDaemons/de.entropytrail.night.plist   # only for uninstall

`bootout` alone pauses the series (the plist stays; re-enable with `bootstrap`).

## Change the queue

Edit `ops/night.env` (`ET_CONDITION`, `ET_PROMPTS`, `ET_SAMPLES`, optional `ET_PROMPT_IDS`,
space-separated). It is read at every start; no launchd reload needed.

## Check

    tail -n 50 ~/Library/Logs/entropytrail-night.log
    tail -n 20 /Volumes/Data/entropytrail/runlog.jsonl     # adjust if ENTROPYTRAIL_DATA differs
    curl -s 127.0.0.1:11434/api/version                    # Ollama must be back up

Manual start for a test: `sudo launchctl kickstart system/de.entropytrail.night`.
Dry run (no side effects): `ET_DRY_RUN=1 bash scripts/night-studio.sh`.

## Watchdog

`--until 06:00` is only checked between tokens. If the run blocks elsewhere (first night:
a macOS privacy prompt for the external volumes), the watchdog terminates it at 06:05
(TERM, KILL after 30 s) and Ollama is restarted. If the log shows a long gap after
"night start", check the Studio screen for a permission dialog; diagnose a hanging
process with `sample <pid>` (main thread in `__open` = blocked file access).
