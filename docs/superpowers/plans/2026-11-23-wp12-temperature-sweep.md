# Temperature Sweep Implementation Plan (Plan 3: WP12)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate, evaluate, rate and analyse the 21-prompt grid at lower temperatures, each as its own condition, and compare it with the T = 1.0 grid.

**Architecture:** No new recorder code: a temperature is a new condition file (new `condition_id`, own seeds, own data folder). What changes is plumbing: the day/night queue must run a *grid* queue in the day window too, and every collect/analysis script that today writes fixed T = 1.0 file names must take the condition as input and write into a per-condition results folder. The comparison across temperatures gets its own short analysis plan, frozen before the first sweep app exists (as D-028 was for the grid).

**Tech Stack:** existing package; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-19-entropytrail-design.md` (§11 WP12, open decision D9), `docs/DECISIONS.md` D-005, D-028–D-036, `docs/RATING-CLAUDE.md` (protocol v1.3).

## Decisions for Robert before Task 1 (defaults = builder's recommendation)

- **T1 — temperatures:** default **0.7 and 0.5** (spec D9 example). 0.7 first; 0.5 only if 0.7 shows a clear change (Task 6 checkpoint), so a null result at 0.7 does not cost four more days.
- **T2 — Claude Chrome ratings:** default **all apps, same protocol v1.3** incl. the novelty calibration pass, because H4/H5-type comparisons need the same outcome on both sides. Cost: one rating agent per prompt as for the grid (21 agents + 21 calibration agents per temperature). Alternative: a stratified sample of 5 apps per prompt (105 per temperature).
- **T3 — comparison plan (frozen in Task 1):** the primary comparisons below.

## Global Constraints

- One temperature = one condition file `configs/conditions/qwen35-9b-bf16-mlx-t0p7.yaml` (`name: qwen35-9b-bf16-mlx-t0p7`, `temperature: 0.7`, all other fields identical to the T = 1.0 file). The T = 1.0 condition and its data are never touched.
- The recorder logs the **untempered** distribution (mlx-lm 0.32.2: log-softmax before the sampler, temperature only inside `categorical_sampling`; checked 2026-11-23), so H, S, forks and all landscape measures are comparable across temperatures; only the sampled paths differ.
- Prompt set v1 unchanged; 20 samples per prompt; seeds from `derive_seed(condition_id, …)` (independent of the T = 1.0 seeds).
- Run windows as now: day 07:15–22:45 (Ollama idle), night 23:00–06:00 (Ollama stopped); a run never starts while another `et run-window` is active.
- Results of a condition other than T = 1.0 go to `results/<condition name>/…`; the existing T = 1.0 files stay where they are.
- **Never edit `scripts/day-studio.sh` or `scripts/night-studio.sh` while that script is running** (bash reads a running script incrementally; an edit can make the waiting instance execute garbage). Day script: edit only when no day run is active (after 22:50, or when the day log shows "day end"); night script: only between 06:05 and 22:59. Always keep the tree clean before 22:45 / 23:00.
- Public repo: commit trailer only `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Queue switch while a fork plan is still running:** setting the grid queue must not be possible while `ET_FORK_PLAN` is set (one queue at a time) — test in Task 2.
2. **Collect scripts called without a condition:** must keep writing the T = 1.0 files exactly as before (no silent overwrite of grid results with sweep results) — test in Task 3.
3. **Mixed conditions in one analysis:** the comparison must refuse to pool apps of different temperatures into one H-test; the per-condition `analyze.py` reads only its own folder — test in Task 3/5.
4. **Condition manifest:** the first sweep run must create a new `condition.json`, never touch the T = 1.0 one — covered by the existing runner guard; verify in Task 4 step 1.

---

### Task 1: Comparison plan and condition file (docs only, before any sweep data)

**Files:** Create `configs/conditions/qwen35-9b-bf16-mlx-t0p7.yaml`; Modify `docs/METHODS.md` (section "Temperature sweep comparison (frozen YYYY-MM-DD)"), `docs/DECISIONS.md` (D-037).

Comparison plan (draft for T3), unit = app, prompt as random effect, `temperature` as a categorical fixed effect (1.0 = reference), Holm across the three primary comparisons per added temperature:
- **C1 renders:** share of apps that render (WP4 render v1) ~ temperature + loose + C(domain), logistic GLM, SE cluster-robust by prompt. Expected: higher at lower T.
- **C2 ensemble spread:** per-prompt screen spread (21 values per temperature) ~ temperature + loose + C(domain), paired by prompt. Expected: lower at lower T.
- **C3 Claude calibrated novelty:** MixedLM novelty_calibrated ~ temperature + loose + C(domain) + (1|prompt). Expected: lower at lower T. (Claude ratings; no human ratings planned for the sweep.)
- Secondary: the H1/H2/H4/H5 analyses of the frozen plan repeated per temperature (H4/H5 with Claude ratings, labelled), fork share and late JS surprise distributions, app length.

- [ ] Write the condition file; check `condition_id` differs from `qwen35-9b-bf16-mlx-1195d88b` (`uv run python -c "from entropytrail.condition import load_condition; print(load_condition('configs/conditions/qwen35-9b-bf16-mlx-t0p7.yaml').condition_id)"`).
- [ ] Write the METHODS section and D-037 (temperatures, rating scope, comparison plan) with Robert's answers to T1–T3; commit `docs: temperature sweep plan frozen (D-037)` **before** Task 4 generates anything.

### Task 2: Day window runs a grid queue too

**When:** only while no day run is active (see Global Constraints) — e.g. after the pilot completes, when the day script exits at once.

**Files:** Modify `scripts/day-studio.sh`, `ops/night.env` (comment), `ops/README.md`; Test `tests/test_day_script.py`.

**Interfaces:** day script runs `et run-window --condition $ET_CONDITION --prompts $ET_PROMPTS --samples $ET_SAMPLES [--prompt-ids …] --until 22:45` when `ET_FORK_PLAN` is empty and `ET_DAY_GRID=1` is set in `ops/night.env`; fork plan wins if both are set and the script logs a configuration warning. Without `ET_DAY_GRID=1` and without a fork plan it does nothing (as now).

- [ ] Failing tests: grid command with `ET_DAY_GRID=1` (contains `--samples 20`, `--until 22:45`, no `--fork-plan`); fork plan + `ET_DAY_GRID=1` → fork command and a warning line; neither → no run line.
- [ ] Implement; `uv run pytest tests/test_day_script.py tests/test_night_script.py -q` → PASS; commit `ops: day window can run a grid queue (WP12)`.

### Task 3: Per-condition results for all collect and analysis scripts

**Files:** Modify `scripts/collect_outcomes.py`, `scripts/collect_measures.py`, `scripts/collect_ratings.py`, `scripts/precheck_apps.py`, `scripts/serve_apps.py`, `scripts/analyze.py`; Test the existing script tests plus one new test per script.

**Interfaces:** every script takes `--root <condition dir>` (already) and derives its output folder with one shared helper `results_dir(root: Path) -> Path` in `src/entropytrail/storage.py`: `results/` for `qwen35-9b-bf16-mlx-1195d88b` (unchanged paths), else `results/<condition_id>/`. `analyze.py` reads its inputs from that folder and writes `<folder>/analysis/`.

- [ ] Failing tests: `results_dir` for the T = 1.0 id returns `results/`; for another id returns `results/<id>/`; `collect_outcomes.main` with a sweep root writes `results/<id>/outcomes.csv` and leaves `results/outcomes.csv` byte-identical.
- [ ] Implement; full suite green; commit `feat: per-condition results folders (WP12)`.

### Task 4: Generate T = 0.7 (operations)

- [ ] After the fork pilot is complete and Robert's D2 decision does not claim the Studio: set `ET_CONDITION=configs/conditions/qwen35-9b-bf16-mlx-t0p7.yaml`, `ET_FORK_PLAN=`, `ET_DAY_GRID=1` in `ops/night.env`; commit; dry-run both scripts.
- [ ] Day smoke: `et generate --condition …t0p7.yaml --prompts prompts/prompts.yaml --prompt-id game-tight-01 --sample 0`; check that `condition.json` of the new id was created and the T = 1.0 one is unchanged (sha256 before/after), `tokens.parquet` H/S look like T = 1.0 values at the same kind of positions.
- [ ] Day and night windows run the queue (420 units, expected ~2 days at ~16 tok/s in both windows). Morning check-ins as for the pilot; JOURNAL entry per day.

### Task 5: Evaluate, rate, analyse T = 0.7

- [ ] `et eval --root /Volumes/Data/entropytrail/<t0p7 id>`; `collect_outcomes`, `collect_measures` into `results/<id>/`.
- [ ] Syntax pre-check and Chrome ratings per `docs/RATING-CLAUDE.md` v1.3 (scope per T2), novelty calibration pass; `collect_ratings` into `results/<id>/ratings-claude.csv`.
- [ ] `analyze.py --root …t0p7` (per-condition H-tests, exploratory, labelled as such for the sweep).
- [ ] Comparison script `scripts/compare_temperatures.py` (C1–C3 of Task 1; test on synthetic two-condition data first: refuses inputs that mix condition ids within one condition's table; temperature coefficient sign and Holm as in `tests_h`). Run; `results/temperature-sweep/summary.json` + `REPORT.md`; JOURNAL; commit.

### Task 6: Checkpoint and T = 0.5

- [ ] Report C1–C3 to Robert. If he goes on: write `configs/conditions/qwen35-9b-bf16-mlx-t0p5.yaml` (as in Task 1, `temperature: 0.5`), repeat Task 4–5 with it and re-run the comparison with three temperatures.
