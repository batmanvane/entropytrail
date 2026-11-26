# Temperature sweep: T = 0.7 vs T = 1.0

Frozen comparison plan: `docs/METHODS.md`, section "Temperature sweep comparison (frozen 2026-11-23)", D-037. Numbers from `summary.json` (this folder, `scripts/compare_temperatures.py`); per-temperature H-tests in `results/analysis/` (T = 1.0) and `results/qwen35-9b-bf16-mlx-t0p7-a31fc218/analysis/` (T = 0.7).

Data: 420 apps per temperature (21 prompts × 20 samples, prompt set v1), WP4 render v1, Claude ratings protocol v1.2 + v1.3 novelty calibration (Claude in Chrome, sonnet; D-030/D-032). No human ratings in the sweep.

## Primary comparisons (one family, Holm m = 3, one-sided)

| | Model | Effect of T = 0.7 | 95 % CI | p (one-sided) | p Holm | Result |
|---|---|---|---|---|---|---|
| C1 apps that render | logit, cluster-robust by prompt | +1.60 log-odds (90.7 % vs 68.6 %) | 1.29 – 1.91 | 8·10⁻²⁵ | 2·10⁻²⁴ | supported |
| C2 ensemble screen spread | OLS, paired by prompt (42 values) | −0.037 (0.123 vs 0.159) | −0.047 – −0.026 | 3·10⁻⁷ | 6·10⁻⁷ | supported |
| C3 calibrated novelty (Claude) | MixedLM, prompt random effect | −0.03 (1.80 vs 1.83) | −0.16 – 0.09 | 0.30 | 0.30 | not supported |

Lowering the temperature from 1.0 to 0.7 makes far more apps run and makes the 20 apps of a prompt look more alike, but Claude's calibrated novelty does not drop. Caveat (frozen plan): novelty is calibrated within each temperature's 20 apps of a prompt, so C3 compares two relative scales; a lower spread of ideas can hide behind a re-centred scale.

## Secondary (descriptive, not in the family)

| measure (mean) | T = 1.0 | T = 0.7 |
|---|---|---|
| app tokens | 4,390 | 3,721 |
| fork share (H ≥ 2 bits) | 0.023 | 0.011 |
| late JS surprise | 0.014 | 0.002 |
| fork surprise | 2.74 | 1.95 |
| path action per token | 0.335 | 0.137 |
| Claude functionality (1–5) | 2.14 | 2.75 |
| Claude ui / ux | 3.07 / 1.97 | 3.26 / 2.45 |
| checklist share (tight prompts) | 0.71 | 0.83 |

Per-temperature H-tests (T = 0.7, secondary): H1 holds (96.4 % of apps reject unimodality vs 97.1 %); H2 holds (loose prompts +0.074 screen spread, CI 0.042 – 0.107, vs +0.098 at T = 1.0); late JS surprise lowers the odds of rendering (p one-sided 0.003); with Claude ratings (exploratory) late JS surprise lowers functionality (p 0.016) and fork surprise raises calibrated novelty (p 0.004), as at T = 1.0.

Rating-process notes: first-pass raters at T = 0.7 wrote each file right after the test (from the 4th prompt on) and game raters were told how to hold keys via dispatched events; calibration depth matched T = 1.0 (about 1–2 minutes per prompt, mostly start screen plus source).
