# Journal

Format: `## YYYY-MM-DD` entries.

## 2026-11-19

Project set up: repository skeleton, environment (uv, Python 3.12, mlx-lm 0.32.2) and documentation skeleton.

## 2026-11-19 (validation and benchmark of the 9B recorder)

First run of the recorder on the real Qwen3.5-9B (bf16, MLX, mlx-lm 0.32.2) on the Mac Studio, with Ollama idle (no model loaded).

- Slow tests (`pytest -m slow`): 3 passed.
- `et bench`, 1000 tokens, 38 prompt tokens: first run was cold (weights read from disk): prefill 156.4 s, decode 12.7 tok/s (not representative). Warm rerun: prefill 3.2 s for 38 tokens (11.8 tok/s, dominated by warm-up; the prompt is too short to measure prefill rate), decode 21.1 tok/s.
- `et validate`, smoke-tool-01, 1500 tokens: **FAILED the threshold.** p99 |dH| = 0.0823 bits, p99 |dS| = 0.1803 bits, max |dH| = 0.2912, max |dS| = 0.3607; same_tokens = true; generate 19.5 tok/s, teacher-forced scoring 36.3 tok/s. Threshold (p99 <= 0.01 bits) exceeded by ~8-18x. No threshold or code was changed; see D-017.
- Smoke app (`et generate`, smoke-tool-01, sample 0, seed 1893739985): 2,731 tokens, finish_reason eos, 136 s (20.1 tok/s), html_extracted true; app.html 10 kB, starts with `<!DOCTYPE html>`, ends with `</html>`, one `<script>`, no external resources (static check only; visual check pending). tokens.parquet schema and meta.json keys match docs/DATA.md. Data root `/Volumes/Data/entropytrail` had to be created.
- Spec section 8 time table replaced by an estimate from these numbers (main grid ~16 h, ~3 nights; forks 42 apps ~51 h, ~8 nights).

## 2026-11-19 (fix round 1: float32 logits head)

- Failure: validation p99 dH 0.0823 / dS 0.1803 bits (see above). Controller probe on the first 800 tokens: teacher forcing with chunk=1 reproduces recorded H and S exactly (diff 0.0), so bookkeeping is correct. With chunk 16/512 the median diff is 2e-6 bits but p99 dH 0.05-0.09 and max dS is exactly 0.180 / 0.361 bits = one bf16 ulp of log-probs at |logp| 8-32 nats. Cause: qwen3_5 `lm_head` outputs bf16 logits and the log-softmax runs in bf16.
- Fix: float32 copy of `lm_head` (`logits_precision: float32` in the condition, new condition_id `qwen35-9b-bf16-mlx-1195d88b`); `et validate` reports a chunk=1 and a chunk=512 block (D-020).
- New numbers: bench (warm) decode 18.9 tok/s. Validate, 1500 tokens: chunk=1 max dH/dS 0.0 (ok); chunk=512 p99 dH 0.0439, p99 dS 0.0829, max dH 0.0947, max dS 0.2543 bits (**still above 0.01, not ok**); same_tokens true; generate 17.0 tok/s. Thresholds not loosened; no further fix tried (D-017).
- Smoke app under the new condition: 4,176 tokens, eos, 241 s (17.3 tok/s), html_extracted true, static check OK, no external resources.
- Spec section 8 table updated (main grid ~28 h / ~4 nights, forks ~90 h / ~13 nights).
- The data of the old condition `qwen35-9b-bf16-mlx-6466454b` under /Volumes/Data/entropytrail is superseded and left in place.

## 2026-11-19 (fix round 2: validation criterion)

Robert decided the criterion (D-021): validation passes iff chunk=1 teacher forcing reproduces recorded H and S within 1e-5 bits max and same_tokens; chunked recomputation is informational (`chunked_noise`). Re-run on the 9B, 1500 tokens, exit 0: chunk=1 max dH/dS 0.0, ok true, same_tokens true; chunked_noise (chunk 512) p99 dH 0.0439, p99 dS 0.0829, max dH 0.0947, max dS 0.2543 bits; generate 15.6 tok/s. Slow tests pass after the head swap.

## 2026-11-19 (final review fixes)

Final review fixes of plan 1 (D-022 to D-026): condition manifest `condition.json` and resume identity guard (prompt and system-prompt hashes, `prompt_ids` in meta.json); mlx-lm version checked at model load; `et generate` / `et run-window` refuse a dirty git tree without `--allow-dirty`, and `validation.json` carries provenance and the weights fingerprint; runlog entry written also on crash (`error`, `git_commit`, `weights_fingerprint`, `stale_moved`), non-done final directories moved aside and redone; night script restarts Ollama only if it stopped it and runs `uv run --frozen --offline`; DATA.md documents every file and key; DECISIONS entries D-002/3/4/6/10/11/16/20 aligned with the spec. Condition fields unchanged, condition_id stays `qwen35-9b-bf16-mlx-1195d88b` (now pinned in a test). The existing validation and smoke data predate these keys; validate/smoke are to be re-run at a clean HEAD.

## 2026-11-19 (clean re-run at 82adefd)

Re-ran on the 9B at a clean tree (git_dirty false, commit 82adefd). `et validate --tokens 1500`: exit 0; chunk=1 max dH/dS 0.0, same_tokens true; chunked_noise (chunk 512) p99 dH 0.0439, p99 dS 0.0829 bits — identical to the earlier run; generate 14.9 tok/s. `condition.json` created for `qwen35-9b-bf16-mlx-1195d88b` (weights_fingerprint f03e7e8f…, mlx 0.32.4, mlx-lm 0.32.2). Smoke unit `smoke-tool-01/000` regenerated: 4176 tokens, eos, 17.0 tok/s, html_extracted true — same length as the earlier dirty-tree run with the same seed, i.e. reproducible across the code changes. The earlier unit was moved to `smoke-tool-01/000.superseded-dirty-6df192d` (not deleted). Visual check of app.html: done by Robert, looks OK.

## 2026-11-20 (first night run)

First night of `de.entropytrail.night` (queue: smoke-tool-01, 20 samples). Started 23:00; the Python process blocked ~50 min in `open()` on a macOS privacy prompt for access to the external volumes (launchd-started processes do not inherit the interactive grant). Robert clicked Allow on the Studio; the run continued 23:49–00:30: 19 units done, 1 skipped (000), 0 aborted, no error, commit a30ba4c, Ollama restarted (0.40.14). Sample 001: 2743 tokens, 15.6 tok/s. Consequence: watchdog added to the night script (D-027).

## 2026-11-20 (WP2 frozen)

Prompt grid v1 (21 prompts) and the analysis plan frozen after Robert's approval (D-028). Night queue now runs the full grid, 20 samples per prompt (420 units); expected ~3–4 nights at ~16 tok/s.

## 2026-11-20 (exploration of the 20 smoke apps)

Exploratory, one tight prompt (smoke-tool-01), 20 samples, 48,122 tokens. 71.7 % of tokens have H < 0.1 bits, 4.5 % have H ≥ 2 bits; the log-H histogram shows a large peak near 0 and a second bump around 0.5–1.5 bits. By section: CSS 43 % of tokens (1.0 % forks), HTML 20 % (1.4 %), JS code 29 % (3.1 %), comments 8 % (34 %) — comments hold 64 % of all forks; the 15 highest-H positions are all comment wordings. Median fork position at 77 % of the file. App length 1,481–4,176 tokens (mean ≈ 2,400), correlation of length with fork share r = 0.71. All 20 are unit converters; 12/20 add an unrequested swap button. All ended with EOS. Consequence: analysis plan amended (D-029).

## 2026-11-20 (quality of the first grid apps)

Chrome ratings (D-030/D-031) of the first prompts: fractions quiz 0/20 fully working (13 broken), times-table trainer 8/20 with functionality ≥ 3, Pythagoras 0/16 usable (8 syntax errors, all drawings wrong). Untruncated T = 1.0 draws 12–42 tokens per app with p < 1 %; in code one such token can break the app. Discussed with Robert: keep the grid at T = 1.0 (the failures are H4 data); a temperature sweep follows after the first DoE (spec D9); generating until 20 syntactically valid apps per prompt is parked as an alternative.

## 2026-11-21 (main grid complete)

All 420 units of prompt set v1 (21 prompts × 20 samples, condition `qwen35-9b-bf16-mlx-1195d88b`) generated: day run 2026-11-20 11:04–22:50 (150), night 23:00–06:00 (142), day run 2026-11-21 07:53–17:27 (128); no errors, two units cut at a deadline and redone. Syntax pre-check and Chrome first-pass ratings (protocol v1.2) run alongside; novelty calibration pass (D-032) follows. First-pass counts of apps with functionality ≥ 3 per prompt: education 15/120, tight games 46/60, loose games 11/60, tight tools 31/60, loose tools 20/60, free-01 10/20, free-02 3/20 (free-03 pending). Well-known tasks (memory, Snake) are reliable and uniform; logic-heavy or open tasks are varied and mostly broken. Strong convergence in open prompts: "Neon …" titles for loose games, 12/17 sliding puzzles for "a puzzle game", to-do lists and Pomodoro timers for "an app of your choice", "Echo/Entropy/Resonance …" audio-particle visualizers for "an app nobody has built before".

## 2026-11-21 (novelty calibration complete)

Calibration pass (D-032, protocol v1.3) done for all 21 prompts: 420/420 files carry `novelty_calibrated` (1: 194, 2: 129, 3: 73, 4: 22, 5: 2; no null). First-pass vs calibrated novelty: r = 0.49, 54 % identical (n = 253 with both values) — confirms the rater drift that motivated the pass. Exploratory, not a test of H5: Spearman ρ(functionality, novelty_calibrated) = −0.25; share of apps with functionality ≥ 3 falls from 43 % (novelty 1) to 28 % (2), 25 % (3) and 9 % (4). The first calibration agent (edu-tight-01) judged from source only; the brief was tightened to require a screenshot per app for all later prompts. `results/ratings-claude.csv` updated.

## 2026-11-22 (WP4: automatic outcomes of the grid)

`et eval` (render v1, SigLIP2 screenshots, Qwen3-Embedding code, fingerprints; D-033, D-034) on all 420 grid apps on the Studio, day run, Ollama idle: 420 done, 0 failed. The first attempt stalled for ~35 min on free-02/012 (a busy loop started by a timer after load blocks Playwright calls without timeout); fixed by rendering each unit in a killable worker process (35 s limit), then resumed (144 skipped, 276 done). Results in `results/outcomes.csv` and `results/ensemble.csv`.

- 288/420 apps render (load without uncaught error), 10 time out or hang. Agreement with the Claude first-pass rating (functionality ≥ 2): 82 %. Disagreements are mostly apps that load but whose core is broken (57 render, functionality 1), and 20 apps with an uncaught load error that still partly work.
- Visual check of 8 random apps (load and interaction screenshots against the Claude `what_it_is` line): all match; the fixed interaction visibly acts (input filled, puzzle tile moved, game started/over).
- Sanity check required by plan 2 (A6), descriptive only, not the H2 test: the most uniform prompts (tip calculator, unit converter, times-table trainer, memory) have the lowest screen spread (0.03–0.08); the free prompts and loose tools the highest (0.25–0.29).

## 2026-11-22 (WP5: fork rollouts smoke test, pilot queued)

Day smoke test of the rollout runner on the 9B (Ollama idle, D-018): one site of the pilot plan (game-tight-01/000, position 1873, a colour constant, H 1.9 bits), 5 rollouts in 1 min 48 s (229–283 tokens each, 7–15 tok/s including prefill of the ~2,000-token prefix), all end with EOS, prefixes identical to the parent up to the forced token. `et eval --forks`: all 5 render; importance 0.010 (noise floor ≈ 0), i.e. the colour choice barely changes the screen. These rollouts are identical to what the pilot run would produce (same code, seeds and forced tokens) and are kept.

Final review of plan 2 (fresh reviewer): fixed the parent-token identity check of the fork runner, the classification of render-worker crashes, and section labels for regex literals and CSS strings (16 of 420 apps change in `results/app-measures.csv`; pilot plan regenerated, free-01/004 sites changed). The night queue now runs `results/forks/plan-pilot.json` (`ET_FORK_PLAN` in `ops/night.env`), expected ~4 nights.

## 2026-11-22 (WP6: interim analysis H1, H2)

Robert decided R1–R3 (D-036, the recommended readings). `scripts/analyze.py` run on the grid: H1 supported (97.1 % of apps reject unimodality; Holm p 4·10⁻²⁵), H2 supported (loose prompts +0.098 screen spread, CI 0.060–0.137; Holm p 1·10⁻⁴). Secondary: late JS surprise lowers the odds of rendering (OR 0.41 per SD), fork surprise raises ensemble rarity slightly; strongest forks are not earlier in loose prompts. Exploratory Claude-rating models point the same way as H4/H5. Details: `results/analysis/REPORT.md`. Figures follow with the H3 run after the fork pilot.

## 2026-11-23 (fork pilot, night 2026-11-22; automatic morning entry)
Night runs: 1 (done 197, aborted 1, errors 0). Rollouts finished since 22:00: 228 (mean 16.0 tok/s); plan total 270/975. `et eval --forks` exit 0, collect exit 0; `results/forks/importance-pilot.csv`: 54 sites with importance.

## 2026-11-24 (fork pilot, night 2026-11-23; automatic morning entry)
Night runs: 1 (done 118, aborted 0, errors 0). Rollouts finished since 22:00: 128 (mean 16.6 tok/s); plan total 975/975. `et eval --forks` exit 0, collect exit 0; `results/forks/importance-pilot.csv`: 195 sites with importance. **fork pilot: complete** — next: H3 run (`scripts/analyze.py`) and the D2 scale decision.

## 2026-11-24 (WP5 pilot complete, H3)

Fork pilot complete: 975/975 rollouts (day and night windows 2026-11-22 to 2026-11-24, no errors; last night ended 03:16). `scripts/analyze.py`: **H3 not supported** — the model with position and section is not better than entropy alone (LRT 6.5, 4 df, p 0.16, Holm 0.49); ρ(H, importance) = −0.14. Fork importance is centred on 0 (SD 0.36); the same-token noise floor (one pair per site) is large and noisy (mean 0.30, 0.13 for Snake, ~0.38 for the loose/open prompts), so the per-site importance is dominated by sampling noise. Details in `results/analysis/REPORT.md`. Fork queue switched off; next: temperature sweep T = 0.7 (D-037), D2 decision by Robert.

## 2026-11-24 (WP12: temperature sweep T = 0.7 started)

Plan 3 Task 2 (day window runs a grid queue, `ET_DAY_GRID`) and Task 4 done: `ops/night.env` now queues condition `qwen35-9b-bf16-mlx-t0p7-a31fc218` (T = 0.7, D-037) in the day (07:15–22:45) and night windows; fork queue off. Smoke unit game-tight-01/000 at T = 0.7: 2,676 tokens, EOS, 17.4 tok/s; mean S 0.072 bits vs 0.176 for the T = 1.0 sample 000 (3 vs 20 tokens with S ≥ 4 bits), mean H 0.108 vs 0.164 bits — H stays on the untempered scale as expected. New `condition.json` created; the T = 1.0 `condition.json` unchanged (sha256 checked). Day run started 08:17. Expected ~2 days for 420 units. Progress is checked by the session check-ins (the 06:30 morning job only handles fork plans and now does nothing).

## 2026-11-24 (D2 decided, 24/7 queue)

Robert chose the fork redesign (D-038: six same-token re-samples; pilot sites extended, 18 further prompts with 2 apps each; H3 primary on the new data) and around-the-clock use of the Studio (D-039). Day and night windows replaced by the queue in `ops/queue.txt`: T = 0.7 sweep (120/420 done at switch time) → its evaluation → `plan-pilot-s6` (780 new rollouts) → `plan-main` (2,268) → fork evaluation. Expected ≈ 1 day for the rest of the sweep and ≈ 5–6 days for the forks.

## 2026-11-25 (WP12: T = 0.7 sweep generated and evaluated)

Queue: T = 0.7 generation complete 08:30 (420/420; overnight run 161 units, no errors), `et eval` 08:30–08:45 (420 done, 0 failed). Collected into `results/qwen35-9b-bf16-mlx-t0p7-a31fc218/` (outcomes, ensemble, app-measures); precheck: 19 apps with JavaScript syntax errors rated automatically (401 left for the Chrome raters). First look vs T = 1.0 (descriptive, not the C1–C3 comparison): renders 90.7 % vs 68.6 %, timeouts 1 vs 10, median 3,576 vs 3,998 tokens, mean screen spread 0.123 vs 0.159 (tight 0.070/0.092, loose 0.144/0.190, open 0.218/0.271). The queue moved on to `plan-pilot-s6` at 08:45. Next: Chrome ratings for T = 0.7 (waiting for Robert).

## 2026-11-25 (WP12: Claude ratings T = 0.7)

Robert approved the Chrome ratings for T = 0.7 (plan 3 T2: all apps, protocol v1.2 first pass + v1.3 novelty calibration, same briefs as T = 1.0 with the data directory switched; app server restarted on the T = 0.7 directory). 21 rating agents (sonnet, one per prompt, sequential) 09:05–11:11 and 21 calibration agents 11:11–11:39; 401 apps tested in Chrome, 19 auto-rated by the syntax pre-check; all 420 have `novelty_calibrated`. Collected into `results/qwen35-9b-bf16-mlx-t0p7-a31fc218/ratings-claude.csv`. Notes: from the 4th prompt on, agents wrote each file right after its test; game agents got a hint for holding keys via dispatched key events (the T = 1.0 raters did the same); calibration depth matched T = 1.0 (≈ 1–2 min per prompt, mostly start screen + source). Descriptive first look vs T = 1.0: functionality mean 2.75 vs 2.14, functionality ≥ 4 in 30.0 % vs 15.5 %, functionality 1 in 15.7 % vs 40.2 %; calibrated novelty 1.80 vs 1.83 (tight 1.39/1.63, loose 1.98/2.01, open 2.45/1.92). The comparison C1–C3 follows with `scripts/compare_temperatures.py`.

## 2026-11-25 (WP12: temperature comparison T = 0.7 vs 1.0)

`scripts/compare_temperatures.py` (frozen C1–C3, Holm m = 3): **C1 supported** — 90.7 % vs 68.6 % of apps render (+1.60 log-odds, Holm p 2·10⁻²⁴); **C2 supported** — ensemble screen spread 0.123 vs 0.159 (−0.037, CI −0.047 to −0.026, Holm p 6·10⁻⁷); **C3 not supported** — Claude calibrated novelty 1.80 vs 1.83 (−0.03, CI −0.16 to 0.09, p 0.30; read with the within-temperature calibration caveat). Per-temperature analysis at T = 0.7: H1 and H2 hold as at T = 1.0 (H2 loose effect +0.074 vs +0.098). Report: `results/temperature-sweep/REPORT.md`. Next (plan 3 Task 6): checkpoint with Robert — T = 0.5 or not.
