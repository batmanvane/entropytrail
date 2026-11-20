# Claude ratings: protocol (v1.1, 2026-11-20)

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

- v1.1 (2026-11-20, after the pilot on edu-tight-01/011 and /000): repeated starts for randomised apps, reading the app source allowed, `novelty: null` for broken apps.
