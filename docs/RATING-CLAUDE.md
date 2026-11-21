# Claude ratings: protocol (v1.3, 2026-11-21)

Every app of the main grid is opened in Chrome by Claude (Claude in Chrome extension), tried out with a fixed test for its prompt type, and rated. These ratings are a **separate outcome** next to the human ratings (spec §5.6, §9). They do not replace them and are never pooled with them (D-030).

## Blindness

The rater sees only the app (served at `http://127.0.0.1:8770/<prompt_id>/<sample>/`) and the prompt text. It does **not** read `tokens.parquet`, `meta.json`, `output.txt`, analysis results or earlier rating files. Apps of one prompt are rated in a shuffled order (seeded by the prompt id), so that "novelty" is not tied to sample numbers.

## Scores (1–5, integers)

| | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| **functionality** | does not load, or the core function is broken | core partly works, major bugs | core works, minor bugs or missing brief items | everything in the brief works, small flaws | everything in the brief works, robust to odd input, no console errors |
| **ui** (visual design) | unstyled or broken layout | styled but cluttered or inconsistent | clean but generic | clean and consistent, some character | polished, consistent, visually appealing |
| **ux** (interaction) | confusing or unusable | usable with significant friction | usable, some friction (unclear states, no feedback) | clear, gives feedback, few rough edges | clear, responsive feedback, pleasant to use, keyboard works where natural |
| **novelty** | standard solution, like most apps of this prompt seen so far | mostly standard, one small twist | some own idea beyond the brief | clearly own idea or unusual feature | surprising idea, not seen in other apps of this prompt |

Novelty is judged relative to the apps of the same prompt seen so far and to the obvious solution of the brief; it does not reward polish (that is ui/ux).

## Automatic pre-check (v1.2)

Before the Chrome test, `scripts/precheck_apps.py` parses each app's inline JavaScript with `node --check`. An app whose JavaScript does not parse (or that has no `app.html`) cannot run; it gets an automatic rating (`rater: "auto-precheck"`, functionality 1, ux 1, ui and novelty `null`) and is not opened in Chrome. All other apps are tested in Chrome as below.

## Checklist keys for the tight prompts

| prompt | keys |
|---|---|
| edu-tight-01 | 10 questions · one at a time · immediate feedback · score at the end |
| edu-tight-02 | tables 1 to 10 · random questions · input field · 60-second timer · correct-answer counter |
| edu-tight-03 | two sliders for a and b · draws right triangle · squares on each side · shows a²+b²=c² with current numbers · numbers correct |
| game-tight-01 | 20×20 grid · arrow keys · grows when eating · speed up every 5 food · score · game over with restart button |
| game-tight-02 | 16 cards (8 pairs) in 4×4 · two cards per move · pairs stay open · move counter · win message |
| game-tight-03 | paddle with left/right arrows · one ball · 5 rows of 10 bricks · 3 lives · score |
| tool-tight-01 | one input field · input-unit selector · other two units shown · updates while typing · values correct |
| tool-tight-02 | 25-min work / 5-min break phases · start · pause · reset · completed-phase counter |
| tool-tight-03 | bill amount · 10/15/20 % buttons · custom tip · number of people · tip and amount per person · values correct |

## Fixed tests by prompt type (2–4 minutes per app)

Before interacting, the page's `alert`, `confirm` and `prompt` are replaced by the server's shim (they log to the console instead of blocking), and the console is cleared.

- **Quiz / trainer (education tight):** answer at least 4 items, at least one right and one wrong on purpose; check feedback; finish or let the timer run out where short; check the final score; try restart.
- **Interactive illustration (Pythagoras):** move both sliders to two settings each; check that the drawing and the numbers update and are correct (e.g. a=3, b=4 → c=5).
- **Games (Snake, memory, Breakout, loose games):** start; play about 30–60 s with keys or clicks; check scoring, losing/winning, restart; check the controls named in the brief.
- **Tools (converter, Pomodoro, tip calculator, loose tools):** enter typical values and check the results by hand (100 cm = 39.37 in = 3.281 ft; tip 15 % of 80 for 4 people = 12 total, 23 per person); try one odd input (empty, negative, text); for timers, start/pause/reset and check the counter logic without waiting the full time.
- **Loose and free prompts:** first find out what the app is meant to do, then exercise its main function as above.

**Randomised content:** if the app generates its content at random (questions, levels, positions), start it at least 3 times, or call its generator function repeatedly from the console, to catch failures that only happen on some starts. The rater may read the app's own source (`app.html`) to check logic such as answer keys; it never reads the recorded token data.

**Broken apps:** if the core does not run, `novelty` is `null` (not observable) and the note says why.

**Checklist (tight prompts):** each concrete requirement of the brief is marked `true`/`false` (e.g. "10 questions", "score at the end").

## Novelty calibration pass (v1.3)

Novelty is relative to the apps of a prompt, and the first-pass raters saw different subsets in different batches (rater drift, e.g. tool-tight-03: novelty 1 throughout in one batch, 2 for similar apps in the next). After all apps of the grid are rated, one calibration agent per prompt opens **all 20 apps of that prompt in one sitting** (screenshot of the start screen and the main view; apps that do not run are judged from their visible idea and source), without reading any rating file, and assigns every app a `novelty_calibrated` score (1–5, same anchors, `null` only if nothing of the idea is observable). It adds two fields to each `rating-claude.json` and leaves all other fields untouched: `novelty_calibrated` and `novelty_calibration` (`{"agent_model": ..., "at": ..., "idea": "one line: the app's core idea"}`). Analyses use `novelty_calibrated`; the first-pass `novelty` stays for comparison (D-032).

## Record

One JSON file per app, next to the app: `/Volumes/Data/entropytrail/<condition_id>/<prompt_id>/<sample>/rating-claude.json`

```json
{
  "rater": "claude-in-chrome",
  "rater_model": "<model id of the rating agent>",
  "protocol": "v1.1",
  "rated_at": "2026-11-20T14:05:00+01:00",
  "functionality": 4,
  "ui": 3,
  "ux": 4,
  "novelty": 2,
  "checklist": {"10 questions": true, "one at a time": true, "immediate feedback": true, "score at the end": true},
  "console_errors": 0,
  "what_it_is": "one line: what the app is",
  "note": "one or two lines: what worked, what failed"
}
```

`checklist` is `{}` for loose and free prompts. A later script collects all files into `results/ratings-claude.csv`.


## Changes

- v1.3 (2026-11-21): novelty calibration pass per prompt after the first pass (D-032).
- v1.2a (2026-11-20, evening): the server clears web storage when a different app is loaded (apps share one origin); ratings before this change may have seen storage keys of other apps (noted in two notes of edu-loose-03, no score affected).
- v1.2 (2026-11-20): automatic syntax pre-check (D-031); fixed checklist keys per tight prompt.
- v1.1 (2026-11-20, after the pilot on edu-tight-01/011 and /000): repeated starts for randomised apps, reading the app source allowed, `novelty: null` for broken apps.
