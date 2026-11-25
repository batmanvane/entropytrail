# Decisions

Format: `## D-NNN · YYYY-MM-DD · title` with Decision / Reason / Alternatives. Every choice made while building is recorded here.

## D-001 · 2026-11-19 · Approach C

**Decision:** Approach C: A first, then B on the same recording core. A is one-shot generation with many samples per prompt, full distributions logged and fork rollouts; B is a crafting loop (draft, render/feedback, revise) with all rounds logged.

**Reason:** Find out whether a pattern exists before building the loop; one shared logger.

**Alternatives:** A alone (cleanest view of the landscape and reachability, but no iterative crafting); B alone (real crafting process, but much more machinery and feedback noise mixes into the model's own decisions); the original plan, observer-only measurement of externally produced apps, rejected as core because it cannot show the generator's landscape.

## D-002 · 2026-11-19 · Primary model Qwen3.5-9B in bf16

**Decision:** Qwen/Qwen3.5-9B at full precision (bf16) is the primary model, frozen for the whole study (spec 4).

**Reason:** Quantization distorts probabilities (spec 4); bf16 (~18 GB) fits the Mac Studio's 32 GB but not the desktop's 16 GB VRAM (spec 8).

**Alternatives:** Qwen3.6-35B-A3B in MLX 4-bit as a later comparison model, a separate condition never mixed with the 9B data (spec 4); Qwen3.5-9B 8-bit on the desktop as an optional separate condition (WP9).

## D-003 · 2026-11-19 · Local only, no Ollama for measurement

**Decision:** All measurement runs are local; Ollama is not used for measurement; the model is run with MLX or PyTorch directly (spec 4; runtime choice in D-016).

**Reason:** Ollama gives no full distributions and no teacher forcing (spec 4).

**Alternatives:** Ollama (rejected for the reason above).

## D-004 · 2026-11-19 · Prompt grid 3x2 plus free cell

**Decision:** 3 domains (education, game, tool) x 2 openness levels (tight, loose) plus a free cell; about 3 prompts per cell, the free cell has ~3 prompts ("Build an HTML app of your choice" + 2 variants), ~21 prompts in total (spec 5.1). All prompts share one fixed system prompt; prompts are versioned in `prompts/prompts.yaml` with stable ids.

**Reason:** Openness is an experimental variable: the hypothesis is that it moves the forks from details (tight) to the idea level (loose/free) (spec 5.1, H2).

**Alternatives:** None recorded in the spec.

## D-005 · 2026-11-19 · 20 samples per prompt, T=1.0, no truncation

**Decision:** 20 samples per prompt, temperature 1.0, no top-p/top-k truncation, max 16,000 tokens.

**Reason:** Samples the model's true distribution, which is what the entropy measures describe.

**Alternatives:** Lower temperature; nucleus sampling.

## D-006 · 2026-11-19 · Thinking mode off

**Decision:** Thinking is disabled via enable_thinking=False in the chat template for the main grid (spec D1).

**Reason:** Thinking tokens are part of the landscape but make runs much longer (spec D1).

**Alternatives:** Thinking on; per the spec D1 proposal, one extra condition with thinking on for the free prompts later.

## D-007 · 2026-11-19 · Top-20 stored, no full logits

**Decision:** Store the top-20 token ids with log-probs; compute H and varentropy from the full distribution at generation time.

**Reason:** Full logits would be about 10 GB per app.

**Alternatives:** Store full logits; top-5.

## D-008 · 2026-11-19 · Fork baseline

**Decision:** Fork experiments are compared against an unforked baseline.

**Reason:** Needed to attribute differences in reachability to the forced token.

**Alternatives:** No baseline.

## D-009 · 2026-11-19 · Analysis plan frozen before WP3

**Decision:** The analysis plan is fixed before work package 3 starts (light pre-registration).

**Reason:** Limits analytic flexibility in an exploratory setting.

**Alternatives:** Fully exploratory analysis.

## D-010 · 2026-11-19 · Studio main series, desktop separate

**Decision:** The Mac Studio produces the main series (all generation and fork experiments of the primary condition). The desktop PC does work that does not mix with the main series (rendering, screenshots, embeddings; optional separate 8-bit condition). The desktop is not a fallback for the Studio's main series; if the Studio is unavailable, the series pauses (spec 8).

**Reason:** One condition = one machine + runtime + precision: logits differ slightly between MLX/Metal and CUDA and between precisions, so a condition is never split across machines (spec 8).

**Alternatives:** Pooling machines (rejected for the reason above).

## D-011 · 2026-11-19 · Run window 23:00-06:00 with Ollama stop/start

**Decision:** Runs happen only between 23:00 and 06:00, started by launchd on the Studio. Ollama on the Studio (`com.robert.ollama`) is stopped at the start of the window and restarted at the end. Jobs are resumable per unit: a unit interrupted at 06:00 is discarded and redone in the next window; finished units are never recomputed (spec 8).

**Reason:** Ollama is stopped to free memory for the 9B model (spec 4, approved by Robert 2026-11-19). The spec gives no reason for the window times.

**Alternatives:** None recorded in the spec.

## D-012 · 2026-11-19 · Public repository

**Decision:** The repository is public; no session trailers or private links in commits.

**Reason:** Reproducibility and transparency.

**Alternatives:** Private repo.

## D-013 · 2026-11-19 · Cloudflare Worker with D1 for ratings

**Decision:** Ratings are collected through a Cloudflare Worker backed by D1.

**Reason:** Cheap, simple, no server to maintain.

**Alternatives:** Self-hosted server; forms.

## D-014 · 2026-11-19 · Raters recruited via Reddit

**Decision:** Human raters are recruited through Reddit.

**Reason:** Reaches interested raters at no cost.

**Alternatives:** Paid panel.

## D-015 · 2026-11-19 · trez63 only as inspiration

**Decision:** trez63.com inspires the kind of app; none of its apps are built, scored or used as data.

**Reason:** Avoids dependence on third-party material.

**Alternatives:** Using its apps as data.

## D-016 · 2026-11-19 · Runtime MLX with mlx-lm 0.32.2 (D6)

**Decision:** Runtime is MLX with mlx-lm pinned exactly to 0.32.2.

**Reason:** mlx-lm 0.32.2 supports the `qwen3_5` architecture (spec 4). Spec D6 requires full-precision logits and hidden-state access. (builder rationale) MLX meets both (float32 logits via D-020; hidden states are accessible from the model's layers), runs natively on Apple Silicon, and the exact pin keeps the condition reproducible.

**Alternatives:** PyTorch-MPS (spec D6).

## D-017 · 2026-11-19 · Validation tolerance exceeded on the 9B model; thresholds unchanged

**Decision:** The validation run on the real model (smoke-tool-01, 1500 tokens) gave p99 |dH| 0.0823 bits, p99 |dS| 0.1803 bits, max |dH| 0.2912, max |dS| 0.3607, same_tokens true, i.e. above the 0.01-bit threshold. The threshold and the code were not changed; the result is reported to Robert for a decision (likely cause to investigate: bf16 numerical differences between incremental decoding with cache and chunked full-sequence teacher forcing).

**Outcome (fix round 1):** Probe showed chunk=1 teacher forcing reproduces the recording exactly, so bookkeeping is correct; chunk>1 differences are bf16 rounding. A float32 lm_head (D-020) reduced but did not remove the gap: chunk=512 p99 dH 0.0439, p99 dS 0.0829, max dH 0.0947, max dS 0.2543 bits, still above 0.01; chunk=1 check is exactly 0.0. Threshold unchanged at that point; resolved by Robert's decision in D-021 (remaining source: bf16 hidden states of the transformer body differ between chunked and single-token kernels).

**Reason:** A tolerance is not loosened silently; the cause needs analysis first.

**Alternatives:** Loosening the threshold to fit the data; recomputing in float32 for scoring.

## D-018 · 2026-11-19 · Validation ran with Ollama idle, not stopped

**Decision:** Stopping Ollama needs sudo, so validation and benchmark ran with Ollama running but idle (`/api/ps` empty, checked before each heavy command, ~86% memory free).

**Reason:** No model was loaded, so memory and speed were not affected.

**Alternatives:** `sudo launchctl bootout system/com.robert.ollama` before the run.

## D-019 · 2026-11-19 · Batched generation is evaluated in the WP3 plan

**Decision:** Batched generation (throughput gain for fork experiments) is not implemented now; it is evaluated in the WP3 plan. The spec section 8 estimate assumes no batching.

**Reason:** Single-sequence throughput (~20 tok/s) already gives a feasible schedule for the main grid.

**Alternatives:** Implementing batching in the foundation plan.

## D-020 · 2026-11-19 · Float32 lm_head (logits_precision: float32)

**Decision:** The condition has a field `logits_precision`; with `float32` the language model's `lm_head` is replaced by a float32 copy and the hidden state is cast to float32 before the matmul, so logits and log-softmax are float32. This changes the condition_id (no study data existed yet). `et validate` now reports two blocks: chunk=1 vs recorder (max <= 1e-5 bits, bookkeeping) and chunk=512 vs recorder (p99 <= 0.01 bits, unchanged). Superseded by D-021: the chunked block is informational only, the p99 <= 0.01 bits criterion no longer applies.

**Reason:** bf16 logits quantize log-probs by one bf16 ulp (0.18-0.36 bits of surprisal for rare tokens at |logp| 8-32 nats), which would contaminate fork and originality measures.

**Cost:** about 4 GB more memory and decode speed about 20 to 17 tok/s.

**Alternatives:** keeping bf16 logits; float32 for the whole model.

## D-021 · 2026-11-19 · Validation criterion (decided by Robert)

**Decision:** Validation passes iff (a) teacher forcing token by token (chunk=1) reproduces the recorded H and S with max abs difference <= 1e-5 bits, and (b) the same seed gives the same token sequence. The chunked recomputation (default chunk 512) is still computed and written to `validation.json` as `chunked_noise`, but is informational only and does not affect the exit code. Measured chunked noise on 1500 tokens of the smoke app: p99 |dH| 0.044, p99 |dS| 0.083, max |dH| 0.095, max |dS| 0.254 bits (bf16 body kernels round differently per chunk size). Rule for later work: any comparison against recorded values recomputes token by token (chunk=1). The earlier "p99 <= 0.01 bits" criterion is replaced.

**Reason:** The recorded values are the exact distribution the sampler used; the chunked differences are noise of the recomputation, not of the recording.

**Alternatives:** Loosening the p99 threshold; running the body in float32.

## D-022 · 2026-11-19 · Condition manifest and resume identity guard

**Decision:** The first run of a condition writes `<data root>/<condition_id>/condition.json` (condition fields, condition_id, weights_fingerprint, system_prompt_version and the sha256 of its text, mlx and mlx-lm versions, created_at). Every later run compares all keys except created_at and aborts before running any unit if one differs. Each unit's `meta.json` also stores `system_prompt_sha256`, `prompt_text_sha256` and the `prompt_ids` fed to the model. When a done unit is skipped on resume, its prompt text hash must equal that of the current prompt, otherwise the run aborts. Units written before this key existed are checked via the sha256 of their stored `prompt.text`. The `Condition` fields are unchanged, so the condition_id stays `qwen35-9b-bf16-mlx-1195d88b` (now pinned in a test).

**Reason:** Units were identified only by condition/prompt/sample, so a changed prompt text, system prompt, weight file or library version would have silently mixed into a condition (final review). (builder rationale) `prompt_set_version` is not compared: adding prompts to the set must not invalidate a condition, and the per-unit text hash catches a changed prompt.

**Alternatives:** Putting fingerprint and versions into `Condition` (would change condition_id); comparing the full meta of every done unit.

## D-023 · 2026-11-19 · mlx-lm version enforced at model load

**Decision:** `load_model` raises before loading if the installed mlx-lm version differs from the condition's `runtime_version`. The mlx version is covered by `condition.json` (D-022).

**Reason:** The runtime version is part of the condition but was not checked (final review).

**Alternatives:** Adding the mlx version to the condition (would change condition_id).

## D-024 · 2026-11-19 · Dirty-tree refusal and provenance in validation.json

**Decision:** `et generate` and `et run-window` refuse to run when `git status --porcelain` is non-empty unless `--allow-dirty` is given; the night run does not pass it. `validation.json` now carries the provenance keys (git_commit, git_dirty, versions, host) and `weights_fingerprint`. (builder rationale) `et validate` and `et bench` are not guarded; validation records `git_dirty` instead.

**Reason:** Data and WP1 evidence must name the exact code that produced them (final review).

**Alternatives:** Only recording `git_dirty` (keeps the data, but its code is not recoverable).

## D-025 · 2026-11-19 · Crash-safe runlog and stale unit directories

**Decision:** `run_units` writes its runlog entry in `finally`, with `error` (null or `"<type>: <message>"`), `git_commit`, `weights_fingerprint` and `stale_moved`, and re-raises the exception. A final unit directory without `status: "done"` is renamed to `<dir>.stale-<YYYYmmdd-HHMMSS>` and the unit is redone. (builder rationale) Renamed rather than deleted, so a possible bug that produced it can still be inspected.

**Reason:** A crash left no runlog trace, and a non-done final directory would have made `write_run` fail on every later night (final review).

**Alternatives:** Deleting the stale directory.

## D-026 · 2026-11-19 · Night script: restart Ollama only if this run stopped it

**Decision:** The EXIT trap of `scripts/night-studio.sh` starts Ollama only after this run has booted it out. The window guard now runs before the mount check, so a missed 23:00 fired after wake (or an unmounted volume) exits without touching Ollama. The run uses `uv run --frozen --offline`.

**Reason:** The skip path would otherwise start a deliberately stopped Ollama; at night uv must neither resolve nor download (final review).

**Alternatives:** Always restarting Ollama on exit (previous behaviour).


## D-027 · 2026-11-20 · Watchdog in the night script

**Decision:** `scripts/night-studio.sh` runs `et run-window` in the background with a watchdog. At 06:05 the watchdog sends TERM to the run (pattern `et run-window --condition`), then KILL after 30 s; the EXIT trap restarts Ollama. Test hooks: `ET_WATCHDOG_SECONDS`, `ET_WATCHDOG_GRACE`, `ET_HANG_RUN` (dry run only).

**Reason:** The first night (2026-11-19) hung for ~50 min in `open()` on a macOS privacy prompt (external volumes, new Python binary). `--until 06:00` is only checked between tokens, so a run blocked elsewhere would have kept Ollama stopped indefinitely. Requested by Robert.

**Alternatives:** Full Disk Access for the Python binary only (does not cover a future new binary); `timeout` (not on macOS by default).

## D-028 · 2026-11-20 · Prompt grid v1 and analysis plan frozen

**Decision:** Prompt set v1 (`prompts/prompts.yaml`: 3 domains × tight/loose × 3 prompts + 3 free prompts = 21) and the analysis plan (METHODS.md, "Analysis plan (frozen 2026-11-20)") are frozen as drafted, including the proposed thresholds: fork = H ≥ 2 bits; late JS surprise = share of tokens in the last third of the JavaScript with S ≥ 4 bits; H3 support requires ρ(H, importance) < 0.5; ≈ 100 rated apps, ≥ 3 ratings each. The smoke prompt set moved to `prompts/prompts-v0-smoke.yaml`. The night queue runs the grid with 20 samples per prompt from 2026-11-20 23:00.

**Reason:** Spec §6 and WP2: the analysis plan must be fixed before the main grid is generated. Approved by Robert.

**Alternatives:** Revising the thresholds before freezing (offered, declined).

## D-029 · 2026-11-20 · Forks counted in code only (analysis plan amendment)

**Decision:** In the frozen analysis plan, a fork is a **code** token with H ≥ 2 bits; comment tokens (JS/CSS `//`, `/* */`, HTML `<!-- -->`) are excluded from fork share, fork surprise, late JS surprise and fork-experiment candidates, and comment forks are reported separately as a secondary measure. H1's dip test still uses all tokens, with a code-only secondary run.

**Reason:** Exploration of the 20 smoke apps (JOURNAL 2026-11-20): comments are 8 % of tokens but hold 64 % of all H ≥ 2 tokens (34 % of comment tokens are forks vs 1.0–3.1 % in CSS/HTML/JS code); the highest-H positions are all comment wordings. Unamended, H3 and H5 would mostly measure comment phrasing rather than decisions in the program. Approved by Robert before any grid data existed.

**Alternatives:** Keep the plan unchanged and add code-only measures as exploratory (rejected: the primary tests would then measure the wrong thing); instruct the model not to write comments (rejected: changes the system prompt and the condition).

## D-030 · 2026-11-20 · Claude ratings of all apps in Chrome

**Decision:** Every app of the main grid (420) is tried out in Chrome by Claude (Claude in Chrome extension) and rated 1–5 on functionality, UI, UX and novelty, with a checklist for tight briefs, console-error count and a short note, following `docs/RATING-CLAUDE.md` (protocol v1). Apps are served by `scripts/serve_apps.py` on 127.0.0.1:8770 with a shim that turns blocking dialogs into console logs; the server exposes only `app.html`. Raters are blind to the recorded token data. Ratings are stored as `rating-claude.json` per unit and kept as a separate outcome, never pooled with the human ratings.

**Reason:** Robert wants quality control of the generated apps and an additional, complete outcome set (the human ratings cover a sample). Novelty added as a fourth criterion on Robert's request.

**Alternatives:** A stratified sample (2 or 5 per prompt) instead of all 420 (offered, declined); automatic checks only (they remain planned as WP4 and complement these ratings).

## D-031 · 2026-11-20 · Keep the grid at temperature 1.0; automatic pre-check before Chrome ratings

**Decision:** The main grid continues unchanged (temperature 1.0, no truncation, frozen plan). Before the Chrome ratings, `scripts/precheck_apps.py` parses each app's JavaScript with `node --check`; apps that do not parse get an automatic rating (functionality 1, ux 1, ui and novelty null) and are not opened in Chrome. Rating protocol v1.2.

**Reason:** The first Chrome batch (edu-tight-01) found no fully working app: 13 of 20 broken on load or first click, 7 playable with major content bugs; syntax errors in 7/20 (fractions quiz), 1/20 (times-table trainer), 6/12 (Pythagoras), 2/20 (smoke converter). Untruncated sampling at T=1.0 regularly draws low-probability tokens, and in code one wrong token breaks the app. For the study these failures are data (H4 can be tested at the exact failure positions), so the condition is kept. The pre-check saves browser work on apps that cannot run at all. Approved by Robert.

**Alternatives:** Lower temperature or top-p truncation for the grid (rejected: changes the frozen condition and hides the true landscape; a lower-temperature comparison condition may follow later as a separate condition_id); rating every app in Chrome regardless (more expensive, same result for unparseable apps).

## D-032 · 2026-11-21 · Novelty calibration pass

**Decision:** After the first-pass Chrome ratings of the whole grid, one calibration agent per prompt rates the novelty of all 20 apps of that prompt in one sitting (`novelty_calibrated`, protocol v1.3). Analyses of novelty (H5) use `novelty_calibrated`; the first-pass `novelty` is kept for comparison.

**Reason:** Novelty is relative to the apps a rater has seen. First-pass raters saw different subsets per batch and drifted (tool-tight-03: novelty 1 for nearly all apps in the first batch, 2 for similar apps in the second). Functionality, UI and UX are tied to absolute anchors and are not affected. Approved by Robert.

**Alternatives:** Keep the first-pass novelty (inconsistent within prompts); re-rate everything in one pass per prompt (far more expensive).

## D-033 · 2026-11-22 · WP4 runs on the Mac Studio

**Decision:** The automatic evaluation (render, screenshots, fingerprints, embeddings) runs on the Mac Studio as a day job (`et eval`), never at the same time as a 9B run. The desktop stays available if Studio time gets short.

**Reason:** Spec §8 assigned WP4 to the desktop so that it does not mix with the main series. Outcome measures are not part of the landscape condition, so the "one condition = one machine" rule does not apply to them. The data, a matching Chromium build (Playwright 1.65.2, Chromium 159.0.8154.6) and the WP5 rollouts are all on the Studio; evaluating there avoids a sync step per rollout batch. Measured: ~0.3 s per app to render, ~1.5 s (median) to 10 s (16k tokens) per code embedding.

**Alternatives:** Desktop with a Studio → desktop sync (spec §8; more moving parts, same results up to float noise of the embedding models).

## D-034 · 2026-11-22 · Outcome measures (render procedure v1, embeddings, outcome distance)

**Decision:**
- *Render v1* (`src/entropytrail/render.py`): headless Chromium, new context per unit, viewport 1024×768, scale 1; `Math.random` replaced by mulberry32 seed 12345; Playwright fake clock paused at 2026-01-01T12:00:00Z, page time advances only by `run_for` (1.5 s before `load.png`); blocking dialogs logged instead; all requests except the app file aborted; 15 s load timeout. Interaction: click the first visible button (else the first canvas), press ArrowRight, ArrowUp, Space, Enter (150 ms page time each), type 42 into the first visible text/number input, 1 s page time, `interact.png`. Screenshots with `animations="disabled"`. `renders` = loaded and no uncaught error up to `load.png`. Each unit is rendered in a worker process; if the page blocks the renderer for more than 35 s (15 s timeout + 20 s grace; first seen on free-02/012, a busy loop after load), the worker is killed and the unit counts as hung (`timeout` true, `renders` false, `hung_killed` true).
- *Embeddings:* screenshots with `google/siglip2-base-patch16-384` (snapshot f775b65a), code with `Qwen/Qwen3-Embedding-0.6B` (snapshot 97b0c614, no instruction prefix, max 16,384 tokens); both L2-normalised, MPS.
- *Screen embedding* of an app = normalised mean of its two screenshot embeddings. *Outcome distance* (spec §5.4) = 1 if exactly one of the two apps renders, else screen cosine distance clipped to [0, 1]. *Ensemble spread / rarity* (frozen plan) = mean pairwise / mean-to-others screen cosine distance **without** the render rule; the render-rule variant is reported alongside.
- **Deviation, labelled:** prompt adherence for tight prompts (H4 secondary, spec §5.5 "hand-written checklist") is taken from the Claude Chrome checklist (D-030) instead of a second, automatic checklist. A Claude-derived measure thus enters a pre-registered secondary test; it is reported as "adherence (Claude checklist)" and never pooled with human ratings.

**Reason:** Screens must be comparable between grid apps and fork rollouts, and same-vs-same rollout distances (the noise floor of fork importance) must come from the model, not the renderer. Check on 30 random grid apps rendered twice: 29 byte-identical screenshot pairs, one pair differed once (CSS animation timing race; four further renders identical). Without the paused clock and disabled animations, animated apps differed on every run.

**Alternatives:** Real time and real randomness (noisy screens); a hand-written automatic adherence checklist per tight prompt (duplicates the Claude checklist, which already exists for all 180 tight apps); CLIP ViT-B/32 (older, 224 px).

## D-035 · 2026-11-22 · Fork experiment pilot (D2)

**Decision:**
- *Pilot* (spec D2 proposal): 3 prompts × 5 apps — `game-tight-01` (tight game), `tool-loose-01` (loose tool), `free-01` (open) — apps drawn with seed 2026 among apps with extracted HTML (`results/forks/plan-pilot.json`: 15 apps, 195 sites, 975 rollouts, ≤ 26 h decoding at 17 tok/s). Scale-up is decided by Robert after the pilot.
- *Sites:* 10 highest-H code tokens (A1 sections html/css/js/text, D-029) greedily at least 20 tokens apart, plus 3 random code tokens (seeded per app) also 20 apart; the last token is never a site. Sites are not required to have H ≥ 2 (in tight apps the top code tokens are often below 2 bits; see R2 in plan 2).
- *Variants:* the top-3 alternatives ≠ actual token from the recorded top-20 (EOS allowed: the app then ends at the prefix), one rollout each; two same-token rollouts.
- *Rollouts:* recorder and condition of the grid; the prefix is prefilled anew per rollout (no KV-cache sharing; the distribution at the site may differ from the recorded one by bf16 chunk noise, D-021, irrelevant because the token is forced); budget `max_tokens − pos − 1` (cap counted from the start of generation); seed per rollout derived from condition, parent, position and variant.
- *Importance:* primary as spec §5.4 (alternatives vs the two same-token rollouts, minus their distance). Secondary `importance_incl_original`: the parent app is added as a third same-token outcome (better noise floor, but sites were chosen with knowledge of the parent's later tokens).
- *Night queue:* `ET_FORK_PLAN` in `ops/night.env` switches the night run to `et run-window --fork-plan`; combined with `ET_PROMPT_IDS` it is a configuration error and the night is skipped without touching Ollama.

**Reason:** Spec §5.4 and D2; the pilot measures real cost and noise floor before committing ~13+ nights.

**Alternatives:** KV-cache sharing across the 5 rollouts of a site (faster prefill, more code, cache copy for the hybrid linear-attention layers); 2 apps per prompt for all 21 prompts directly (spec §8; decided after the pilot).

## D-036 · 2026-11-22 · Readings of the frozen analysis plan (R1–R3) and model coding

**Decision (Robert, 2026-11-22, before any confirmatory test was computed):**
- **R1 (H1):** support is decided by the frozen rule "≥ 80 % of apps reject unimodality" (`support_frozen`). For the Holm family H1 enters with the one-sided binomial p-value of the rejecting share against 0.80 (H0: share ≤ 0.80). Both are reported; a disagreement between them is stated, not resolved.
- **R2 (H3):** all fork sites of the fork plans enter the H3 test (10 top-H code tokens and 3 random code tokens per app, D-035). Sensitivity runs: sites with H ≥ 2 bits only, and sites whose parent app is not truncated.
- **R3 (H4):** late JS surprise = share of tokens with S ≥ 4 bits among the last third (by count, index ≥ ⌈2n/3⌉) of the app's JavaScript **code** tokens, JS comments removed before the cut.
- **Model coding (all models):** `loose` dummy (1 = loose prompt) plus `C(domain)` with levels education/game/tool/free, all 21 prompts, because openness `open` and domain `free` coincide. Holm always with m = 5; until all five p-values exist results are flagged interim and the multiplier uses the rank among the available p-values (never smaller than the final adjustment).
- **Fixed secondaries:** H4 renders → logistic GLM with cluster-robust SE by prompt; H4 adherence from the Claude checklist (D-034 deviation); H5 ensemble rarity as outcome with fork surprise as predictor; H5 alternative predictor path action per token. Sensitivity of every primary test without truncated apps (H2: spread recomputed from the non-truncated apps of each prompt).

**Reason:** The frozen plan (D-028, D-029) leaves these details open; they change the numbers and must be fixed before results exist (Claude ratings already exist). Recommendations and alternatives in plan 2 (Part C) were explained to Robert, who chose the recommended option in all three cases.

**Alternatives:** R1 — support only if the binomial test survives Holm (stricter, changes the frozen rule), or H1 outside the Holm family; R2 — only sites with H ≥ 2 bits (literal "fork", narrow H range, tight apps drop out); R3 — cut the last third over JS code and comment tokens, then drop comments.

## D-037 · 2026-11-23 · Temperature sweep (WP12): temperatures, rating scope, comparison plan

**Decision (Robert, 2026-11-23):**
- **T1:** first sweep condition T = 0.7 (`configs/conditions/qwen35-9b-bf16-mlx-t0p7.yaml`, condition_id `qwen35-9b-bf16-mlx-t0p7-a31fc218`, all other fields as the T = 1.0 condition). T = 0.5 only after the T = 0.7 comparison, if it shows a clear change.
- **T2:** all 420 apps per temperature get the Claude Chrome ratings (protocol v1.3) including the novelty calibration pass.
- **T3:** comparison plan as frozen in METHODS.md "Temperature sweep comparison (frozen 2026-11-23)": C1 renders, C2 ensemble spread, C3 calibrated novelty, Holm across the three.
- The sweep starts after the fork pilot (D-035) is complete; it uses the day and night windows.

**Reason:** Spec D9 / WP12. At T = 1.0, 132 of 420 apps do not render and late improbable JS tokens go with apps that fail (interim WP6); the sweep measures the trade-off between running apps and variety. 0.7 first so that a null result does not cost a second sweep. Seeds are derived from the new condition_id, i.e. independent of the T = 1.0 samples. Verified 2026-11-23: mlx-lm 0.32.2 computes the log-softmax before the sampler and `categorical_sampling` only scales the logits by 1/T, so the recorded H and S are untempered.

**Alternatives:** 0.7 and 0.5 at once (twice the compute before knowing whether 0.7 changes anything); a rating sample of 5 apps per prompt (cheaper, but C3 and the per-temperature H4/H5 would rest on 105 apps); top-p truncation instead of lower temperature (changes the distribution's support; rejected in D-031 for the main grid).

## D-038 · 2026-11-24 · Fork experiments redesigned: six same-token re-samples; H3 on the new data (amendment)

**Decision (Robert, 2026-11-24, D2 option 1, before any data of the new design existed):**
- **Baseline:** 6 same-token re-samples per site (`same1`–`same6`) instead of 2, so the noise floor is a mean over 15 pairs instead of one; 3 alternatives as before. Fork importance and its definition (spec §5.4) are unchanged.
- **Pilot sites reused:** `results/forks/plan-pilot-s6.json` = the 195 pilot sites with `same3`–`same6` added (780 new rollouts); the 975 pilot rollouts are reused unchanged (seeds depend on site and variant name only).
- **Scale:** `results/forks/plan-main.json` = 2 apps (seed 2027) for each of the other 18 prompts, 5 top-H + 2 random code sites per app (min spacing 20), 3 alternatives + 6 re-samples per site: 36 apps, 252 sites, 2,268 rollouts.
- **H3 amendment:** the primary H3 test (frozen model, criterion and Holm slot unchanged; readings of D-036) is computed on the redesigned data: all sites of `plan-pilot-s6` and `plan-main`, importance with six same-token re-samples. The pilot result with two re-samples (2026-11-24: not supported, LRT p 0.16, ρ −0.14) stays reported as the pilot.

**Reason:** In the pilot the noise floor rested on one pair per site and was large and erratic (mean 0.30; whenever one of the two re-samples breaks, the pair distance is 1), so per-site importance was dominated by sampling noise and H3 could not separate "the choice mattered" from noise. More apps with the same design would mostly measure more noise; three prompts are too few for a test about entropy vs consequence across prompt types.

**Alternatives:** scale-up with the pilot design (spec §8, 2 apps per prompt); stop the forks; only extend the pilot sites (≈ 1 day); one app per prompt in the main plan (≈ 3 days in total). The redesigned data as a secondary analysis only, H3 staying "not supported" (rejected: the pilot was meant as a pilot, spec D2).

## D-039 · 2026-11-24 · 24/7 job queue instead of day and night windows

**Decision (Robert, 2026-11-24: the Studio may be used around the clock):** a user LaunchAgent `de.entropytrail.queue` (KeepAlive) runs `scripts/queue_runner.py`, which works through `ops/queue.txt` in order without time windows: T = 0.7 sweep generation, its `et eval`, fork plans `plan-pilot-s6` and `plan-main`, `et eval --forks` of the T = 1.0 condition. Each job repeats until nothing is left (finished units are never recomputed); a job never starts while another `et run-window` runs; a job without a new finished unit for 2 hours is killed and restarted; a refused attempt (dirty tree) is retried every 10 min. Night (`ET_NIGHT_ENABLED=0`) and day (`ET_DAY_GRID=0`, no fork plan) scripts stay installed but inert; the night script additionally refuses to run while another 9B run is active. Ollama stays running and idle (D-018).

**Reason:** No windows needed; one queue is simpler and removes the hand-over at 22:45/23:00 and the morning switch. Evaluation runs as a queue job between generation jobs, so it never shares memory with the 9B.

**Alternatives:** keep day + night windows (more moving parts, gaps); stop Ollama permanently (needs root; not needed while it is idle). Removing the night LaunchDaemon needs sudo: `sudo launchctl bootout system/de.entropytrail.night` (optional).

## D-040 · 2026-11-25 · Temperature sweep continues with T = 0.5

**Decision (Robert, 2026-11-25, plan 3 Task 6 checkpoint):** after the T = 0.7 comparison (C1 and C2 supported, C3 not; `results/temperature-sweep/REPORT.md`) the sweep continues with T = 0.5. Condition `configs/conditions/qwen35-9b-bf16-mlx-t0p5.yaml` (identical to the T = 0.7 file except name and `temperature: 0.5`; id `qwen35-9b-bf16-mlx-t0p5-c9bce32f`). Same protocol as T = 0.7 (D-037): 420 apps, WP4 eval, Claude ratings v1.2 + v1.3 calibration for all apps, comparison C1–C3 against T = 1.0 as a second family (Holm m = 3), then the comparison rerun with three temperatures.

**Scheduling:** appended to `ops/queue.txt` after the fork jobs (D-038), so the forks are not delayed: grid generation (≈ 1.5 days at the T = 0.7 pace) and its evaluation.

## D-041 · 2026-11-25 · Cross-temperature blind novelty pass (exploratory)

**Decision (Robert, 2026-11-25):** after the T = 0.5 Claude ratings, one more novelty pass scores **all** apps of a prompt from all three temperatures (60 per prompt) on one common scale, blind to temperature: the pool is shuffled per prompt (seed 41) and served under blind ids by `scripts/serve_apps.py --port 8771 --blind-key /Volumes/Data/entropytrail/_novelty-cross/key.json`; one sonnet agent per prompt (`docs/novelty-cross-brief.md`) surveys all apps (idea line + screenshot each), then scores 1–5 relative to the whole pool; scores are kept apart from the per-app rating files. Analysis (`scripts/novelty_cross.py collect`): MixedLM `novelty_cross ~ temperature (ref 1.0) + loose + C(domain) + (1|prompt)`, one-sided p for lower novelty at lower T, reported with means per temperature.

**Status:** exploratory. It was defined after the frozen C3 result (calibrated novelty within temperature, D-037) was seen, so it is not part of any Holm family; C3 stays the primary comparison and is reported unchanged.

**Reason:** the within-temperature calibration makes C3 compare two relative scales; a common scale shows whether lower temperatures produce fewer distinct ideas. Running once after T = 0.5 keeps one scale for all three temperatures.

**Cost and billing:** ≈ 21 agents × 5–10 min; runs inside Claude Code on Robert's Claude subscription (the shell's `ANTHROPIC_API_KEY` is rejected in Claude Code), not the API.
