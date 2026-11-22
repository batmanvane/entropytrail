# Interim analysis (2026-11-22)

Condition `qwen35-9b-bf16-mlx-1195d88b`, prompt grid v1, 420 apps. Frozen plan: METHODS.md (D-028, D-029); readings R1–R3 and model coding: D-036. Numbers from `summary.json` (provenance there). **Interim:** H3 waits for the fork pilot (WP5), H4/H5 primary for human ratings (WP7). Holm uses m = 5; adjusted p-values can only grow as the family fills.

## Primary tests

| | criterion (frozen) | result | p (one-sided) | Holm p (interim) |
|---|---|---|---|---|
| **H1** bimodal entropy | ≥ 80 % of apps reject unimodality (dip test on log10(H + 0.001)) | **97.1 %** of 420 apps reject → supported | 7·10⁻²⁶ (binomial vs 0.80, R1) | 4·10⁻²⁵ |
| **H2** openness → spread | loose > tight in screen spread (OLS spread ~ loose + domain, 21 prompts) | loose +0.098 (95 % CI 0.060–0.137) on a spread scale where tight prompts lie at 0.03–0.15 → supported | 3·10⁻⁵ | 1·10⁻⁴ |
| H3 entropy ≠ consequence | larger model better (LRT) and ρ(H, importance) < 0.5 | waiting for the fork pilot | | |
| H4 usefulness | human usefulness ~ late JS surprise, negative | waiting for human ratings | | |
| H5 novelty | human novelty ~ fork surprise, positive | waiting for human ratings | | |

Sensitivity without the 1 truncated app: H1 97.1 % (419 apps); H2 +0.098, p 3·10⁻⁵. Code tokens only (H1 secondary): 93.6 % reject.

## Secondary tests (fixed in D-036)

- **H2, fork position:** median relative position of the 10 highest-H forks does not move earlier for loose prompts (+0.02, two-sided p 0.60). The idea-level choice of loose prompts shows up in the outcome spread, not in where the strongest forks sit.
- **H4, renders:** apps with more late JS surprise render less often: odds ratio 0.41 per SD of late JS surprise (logit, cluster-robust by prompt, one-sided p 0.002).
- **H5, ensemble rarity as outcome:** apps with higher fork surprise are rarer within their prompt: +0.006 screen distance per SD of fork surprise (one-sided p 0.0007). Small effect.

## Exploratory (Claude ratings, D-030 — not human ratings, not in the Holm family)

- Claude functionality ~ late JS surprise: −0.31 rating points per SD (n 357 apps rated in Chrome, p 2·10⁻⁶).
- Claude calibrated novelty ~ fork surprise: +0.11 points per SD (n 354, one-sided p 0.012).

These point the same way as H4 and H5 but do not test them; the primary outcomes are the human ratings.

## Plain reading so far

The landscape is as expected strongly two-peaked: almost every app is mostly near-certain tokens with a minority of open positions (H1). Loose briefs produce visibly more varied apps than tight ones (H2). Improbable tokens late in the JavaScript go with apps that fail to run (H4 secondary), and surprising choices at forks go with apps that look different from their siblings (H5 secondary) — whether that translates into what people find useful or new is open until the human ratings exist.
