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

## Fork queue (WP5)

Set `ET_FORK_PLAN=results/forks/plan-pilot.json` (path relative to the repo) in
`ops/night.env` to run the rollouts of a fork plan instead of the grid; `ET_PROMPT_IDS`
must then be empty. The plan file must be committed (the run refuses a dirty tree).
Progress: `grep '"kind": "forks"' /Volumes/Data/entropytrail/runlog.jsonl | tail`.
Set `ET_FORK_PLAN=` again to return to the grid queue.

## Morning job after fork nights

A user LaunchAgent `de.entropytrail.morning` runs `scripts/morning_forks.py` daily at 06:30
(after the night's 06:00 end and the 06:05 watchdog). While `ET_FORK_PLAN` is set it runs
`et eval --forks` for the plan's prompts, rewrites `results/forks/importance-<plan>.csv`,
appends an automatic entry to `docs/JOURNAL.md` and commits exactly these two files (no
push). It skips if a night run is still active, does not commit if either file had
uncommitted changes, and stops once it has written "fork <plan>: complete". It never
commits other files, so a dirty tree still blocks the night (by design). Log:
`~/Library/Logs/entropytrail-morning.log`. Test without writing: `ET_MORNING_DRY=1 uv run
--group eval --group analysis python scripts/morning_forks.py`.

    cp ops/launchd/de.entropytrail.morning.plist ~/Library/LaunchAgents/
    launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/de.entropytrail.morning.plist
    # remove: launchctl bootout gui/$(id -u)/de.entropytrail.morning

## Day runs (fork plans or a grid queue)

A user LaunchAgent `de.entropytrail.day` runs `scripts/day-studio.sh` daily at 07:15 (after
the 06:30 morning job). While `ET_FORK_PLAN` is set it runs `et run-window --fork-plan …
--until 22:45`; otherwise, with `ET_DAY_GRID=1`, it runs the grid queue of `ops/night.env`
(`--samples`, optional `--prompt-ids`) until 22:45 with Ollama left running but idle (D-018); loading a large Ollama model during
the day competes for memory with the 9B (~23 GB). It skips outside 07:00-21:59, when neither
queue is set, or when another `et run-window` is active, and its own watchdog kills the
run's process tree at 22:50, so it never overlaps the 23:00 night run. Unfinished rollouts
are discarded and redone, as at night. Log: `~/Library/Logs/entropytrail-day.log`.

    cp ops/launchd/de.entropytrail.day.plist ~/Library/LaunchAgents/
    launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/de.entropytrail.day.plist
    launchctl kickstart gui/$(id -u)/de.entropytrail.day      # start now
    # pause: launchctl bootout gui/$(id -u)/de.entropytrail.day

## Watchdog

`--until 06:00` is only checked between tokens. If the run blocks elsewhere (first night:
a macOS privacy prompt for the external volumes), the watchdog terminates it at 06:05
(TERM, KILL after 30 s) and Ollama is restarted. If the log shows a long gap after
"night start", check the Studio screen for a permission dialog; diagnose a hanging
process with `sample <pid>` (main thread in `__open` = blocked file access).
