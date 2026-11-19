# Night scheduling (Mac Studio)

A root LaunchDaemon `de.entropytrail.night` runs `scripts/night-studio.sh` daily at 23:00.
The script stops the system Ollama daemon (`com.robert.ollama`), runs
`et run-window ... --until 06:00` as user robert, and restarts Ollama on every
exit path (EXIT trap), including failures and unmounted volumes.

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
