# Analysis plan – DRAFT (to be frozen before WP3)

Status: draft for Robert's review, 2026-11-19. Once approved it is moved into `docs/METHODS.md` as "Analysis plan (frozen YYYY-MM-DD)" and committed **before** the main grid is generated (spec §6, light pre-registration). Items marked **[decide]** need Robert's call.

## Data and units

- Condition `qwen35-9b-bf16-mlx-1195d88b`, prompt grid v1 (21 prompts), 20 samples per prompt → 420 apps.
- Unit of analysis: the app (one sample). Prompts are a random effect; domain (education / game / tool / free) and openness (tight / loose / open) are fixed effects.
- Token-level values are the recorded ones (exact sampling distribution, D-021). Any recomputation that is compared with them uses chunk=1.
- Only the app part counts: tokens from the start of the extracted HTML to its end. Apps without extracted HTML are kept in the outcome tables (as "no HTML") but excluded from token-level measures.
- Truncated runs (`finish_reason = max_tokens`) are included and flagged; a sensitivity analysis repeats each primary test without them.

## Definitions used below

| Name | Definition |
|---|---|
| fork | token with H ≥ 2 bits |
| fork share | forks / tokens of the app |
| relative position | token index / app length (0 = start, 1 = end) |
| ensemble spread | mean pairwise distance between the 20 apps of a prompt (screenshot embedding, cosine) |
| ensemble rarity | mean distance of an app to the other 19 apps of its prompt |
| fork importance | spec §5.4: alternative-vs-same-token outcome distance minus the same-token noise floor |
| late JS surprise | share of tokens in the last third of the app's JavaScript with S ≥ 4 bits |
| fork surprise | mean S over the app's forks |

Thresholds (2 bits, 4 bits, last third) are fixed here so they cannot be tuned after seeing data. **[decide]** whether to keep these values.

## Primary tests (one per hypothesis, Holm-corrected across H1–H5, α = 0.05)

**H1 – bimodality.** Per app: Hartigan's dip test on log10(H + 10⁻³). Support if the dip test rejects unimodality (p < 0.05) in ≥ 80 % of apps. Reported alongside: share of tokens with H < 0.1 bits and fork share, by domain and openness.

**H2 – openness moves forks.** Outcome: ensemble spread per prompt (21 values). Model: linear regression spread ~ openness + domain. Support if loose > tight (one-sided). Secondary (not part of the Holm family): median relative position of the 10 highest-H forks per app, mixed model ~ openness + domain + (1 | prompt); expected earlier for loose/open.

**H3 – entropy ≠ consequence.** Data: fork experiments (subset, spec §5.4). Outcome: fork importance per fork. Model comparison: importance ~ H vs importance ~ H + relative position + section, both with (1 | app). Support if the larger model is better (likelihood-ratio test) **and** Spearman ρ(H, importance) < 0.5. **[decide]** the 0.5 cut-off.

**H4 – usefulness.** Outcome: mean human usefulness rating per app (only apps with ≥ 3 ratings). Primary predictor: late JS surprise. Model: rating ~ late JS surprise + domain + openness + (1 | prompt). Support if the coefficient is negative (one-sided). Secondary: renders yes/no and prompt adherence (tight prompts) with the same predictor.

**H5 – novelty.** Outcome: mean human novelty rating per app (≥ 3 ratings). Primary predictor: fork surprise. Model: rating ~ fork surprise + domain + openness + (1 | prompt). Support if positive (one-sided). Secondary: ensemble rarity as outcome; path action (Σ S, per token) as alternative predictor.

## Rules

- Every analysis not listed above is exploratory and labelled as such in notebooks and reports.
- Null results are reported with effect sizes and confidence intervals.
- Changes after freezing get a dated entry in DECISIONS.md with the reason, and the original stays visible in git.
- Ratings: apps shown in random order, blind (spec §9); the rating sample is stratified by prompt (≈ 5 apps per prompt → ≈ 100 apps, D4). **[decide]** sample size if Reddit response is low.
