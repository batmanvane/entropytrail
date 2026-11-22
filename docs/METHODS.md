# Methods

All information quantities are in bits (log base 2). Let p be the full next-token distribution at step t and x_t the sampled token.

- **Entropy H** = −Σ p log₂ p
- **Surprisal S** = −log₂ p(x_t)
- **Varentropy** = Var_p[−log₂ p] = Σ p (log₂ p)² − H²
- **Margin** = log₂ p(top1) − log₂ p(top2)

## Implementation notes (`src/entropytrail/stats.py`)

- Logits and log-softmax are computed in float32 from a float32 copy of `lm_head` (condition field `logits_precision: float32`); the transformer body stays bf16.
- Input is the natural-log probability vector from `generate_step` (log-softmax of the logits). With the float32 head it is already float32 (the cast to float32 in `stats.py` only matters for a condition with `logits_precision: native`, where log-probs are in the model dtype bf16); it is converted to bits by dividing by ln 2, and all four quantities are computed on device in float32.
- Entries with p = 0 (log-prob = -inf) contribute exactly 0 to H and to the second moment (`where(p > 0, ..., 0)`), so no NaNs arise. Varentropy is clamped at 0 against rounding.
- Surprisal S uses the log-prob of the token actually emitted (or, in teacher forcing, the forced token).
- Margin uses the two largest log2 probabilities (top-1 minus top-2); 0 if the vocabulary has a single entry.
- Top-k: k = min(20, V), found with `argpartition` and then sorted descending; stored as token ids (int32) and log2 probabilities (float32).

## Precision

Recorded values are the exact sampling distribution: the transformer body runs in bf16, the head (float32 copy of `lm_head`) and the log-softmax in float32. A chunked teacher-forced recomputation (chunk > 1) differs from the recorded values because the bf16 body kernels round differently for different chunk sizes; measured on 1500 tokens of the smoke app (chunk 512): p99 |dH| 0.044, p99 |dS| 0.083, max |dH| 0.095, max |dS| 0.254 bits. Compare against recorded values only by recomputing token by token (chunk = 1), which reproduces them exactly (decision D-021).

## Outcome measures (WP4, defined 2026-11-22, before any fork rollout existed)

See D-034 for the render procedure and models. With e_load, e_interact the L2-normalised image embeddings of an app's two screenshots:

- **screen embedding** s = (e_load + e_interact) / ‖e_load + e_interact‖
- **screen distance** d_s(a, b) = clip(1 − s_a·s_b, 0, 1)
- **outcome distance** d(a, b) = 1 if exactly one of a, b renders, else d_s(a, b) (spec §5.4; used for fork importance)
- **ensemble spread** of a prompt = mean of d_s over all pairs of its apps; **ensemble rarity** of an app = mean d_s to the other apps of its prompt (no render rule, as worded in the frozen plan; the render-rule variant is reported alongside)
- code and fingerprint spread/rarity: the same with the cosine distance of code embeddings and with `fingerprint_distance` (mean of tag-histogram cosine distance and Jaccard distances of JS function names and CSS colours)

## Analysis plan (frozen 2026-11-20)

Approved by Robert on 2026-11-20 (D-028); amended the same day before any grid data existed (D-029: forks counted in code only, comments reported separately). Committed before the main grid (prompt set v1) was generated — light pre-registration per spec §6. Changes after this date need a dated DECISIONS.md entry; the original stays visible in git history.

### Data and units

- Condition `qwen35-9b-bf16-mlx-1195d88b`, prompt grid v1 (21 prompts), 20 samples per prompt → 420 apps.
- Unit of analysis: the app (one sample). Prompts are a random effect; domain (education / game / tool / free) and openness (tight / loose / open) are fixed effects.
- Token-level values are the recorded ones (exact sampling distribution, D-021). Any recomputation that is compared with them uses chunk=1.
- Only the app part counts: tokens from the start of the extracted HTML to its end. Apps without extracted HTML are kept in the outcome tables (as "no HTML") but excluded from token-level measures.
- Truncated runs (`finish_reason = max_tokens`) are included and flagged; a sensitivity analysis repeats each primary test without them.

### Definitions used below

| Name | Definition |
|---|---|
| comment token | token whose first character lies inside a comment: JS/CSS `// …` and `/* … */`, HTML `<!-- … -->`; detected by a scan that ignores comment markers inside string literals and URLs *(amendment 2026-11-20, D-029)* |
| code token | any token of the app that is not a comment token |
| fork | **code** token with H ≥ 2 bits *(amendment 2026-11-20, D-029; originally any token)* |
| comment fork | comment token with H ≥ 2 bits (secondary measure, reported separately) |
| fork share | forks / code tokens of the app |
| relative position | token index / app length (0 = start, 1 = end) |
| ensemble spread | mean pairwise distance between the 20 apps of a prompt (screenshot embedding, cosine) |
| ensemble rarity | mean distance of an app to the other 19 apps of its prompt |
| fork importance | spec §5.4: alternative-vs-same-token outcome distance minus the same-token noise floor |
| late JS surprise | share of code tokens in the last third of the app's JavaScript with S ≥ 4 bits (comments excluded, D-029) |
| fork surprise | mean S over the app's forks |

Thresholds (2 bits, 4 bits, last third) are fixed here so they cannot be tuned after seeing data.

### Primary tests (one per hypothesis, Holm-corrected across H1–H5, α = 0.05)

**H1 – bimodality.** Per app: Hartigan's dip test on log10(H + 10⁻³). Support if the dip test rejects unimodality (p < 0.05) in ≥ 80 % of apps. Reported alongside: share of tokens with H < 0.1 bits, fork share and comment-fork share, by domain and openness. The dip test runs on all tokens (H1 concerns the whole distribution); a secondary run uses code tokens only.

**H2 – openness moves forks.** Outcome: ensemble spread per prompt (21 values). Model: linear regression spread ~ openness + domain. Support if loose > tight (one-sided). Secondary (not part of the Holm family): median relative position of the 10 highest-H forks per app, mixed model ~ openness + domain + (1 | prompt); expected earlier for loose/open.

**H3 – entropy ≠ consequence.** Data: fork experiments (subset, spec §5.4), forks selected among code tokens (D-029). Outcome: fork importance per fork. Model comparison: importance ~ H vs importance ~ H + relative position + section, both with (1 | app). Support if the larger model is better (likelihood-ratio test) **and** Spearman ρ(H, importance) < 0.5.

**H4 – usefulness.** Outcome: mean human usefulness rating per app (only apps with ≥ 3 ratings). Primary predictor: late JS surprise. Model: rating ~ late JS surprise + domain + openness + (1 | prompt). Support if the coefficient is negative (one-sided). Secondary: renders yes/no and prompt adherence (tight prompts) with the same predictor.

**H5 – novelty.** Outcome: mean human novelty rating per app (≥ 3 ratings). Primary predictor: fork surprise. Model: rating ~ fork surprise + domain + openness + (1 | prompt). Support if positive (one-sided). Secondary: ensemble rarity as outcome; path action (Σ S, per token) as alternative predictor.

### Rules

- Every analysis not listed above is exploratory and labelled as such in notebooks and reports.
- Null results are reported with effect sizes and confidence intervals.
- Changes after freezing get a dated entry in DECISIONS.md with the reason, and the original stays visible in git.
- Ratings: apps shown in random order, blind (spec §9); the rating sample is stratified by prompt (≈ 5 apps per prompt → ≈ 100 apps, D4). If fewer than 3 ratings per app are reached, the sample is reduced stratified by prompt rather than relaxing the ≥ 3 rule; the reduction is logged in DECISIONS.md.

## Implementation notes (WP6, 2026-11-22)

Code: `src/entropytrail/analysis/measures.py`; table: `results/app-measures.csv` (one row per app).

- **App part:** tokens whose first character lies in the extracted HTML (`extract.app_span`); `rel_pos` = index among app tokens / number of app tokens.
- **Sections:** `sections.token_sections(..., detail=True)`; code = html, css, js, text; comment = js-comment, css-comment, html-comment (D-029 rule: label of the token's first character; comment markers inside JS strings are code; regex literals are not detected).
- **late JS surprise** (reading R3, pending Robert's confirmation before any test uses it): take the JS **code** tokens (label `js`, JS comments dropped first) in order, keep the last third by count (index ≥ ⌈2n/3⌉), share with S ≥ 4 bits; undefined (NaN) if the app has fewer than 3 JS code tokens (0 of the 420 grid apps).
- **fork surprise:** mean S over forks; NaN if the app has no fork. **median_relpos_top10_forks:** median `rel_pos` of the (up to) 10 forks with the highest H.
- **path action:** Σ S over app tokens; also per token. **H per char / per line:** Σ H over app tokens divided by characters / lines of `app.html`.
- Check: the 20 smoke apps give 71.7 % of app tokens with H < 0.1 bits and 4.5 % with H ≥ 2 bits, as in JOURNAL 2026-11-20.
