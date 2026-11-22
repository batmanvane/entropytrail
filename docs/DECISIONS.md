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
- *Render v1* (`src/entropytrail/render.py`): headless Chromium, new context per unit, viewport 1024×768, scale 1; `Math.random` replaced by mulberry32 seed 12345; Playwright fake clock paused at 2026-01-01T12:00:00Z, page time advances only by `run_for` (1.5 s before `load.png`); blocking dialogs logged instead; all requests except the app file aborted; 15 s load timeout. Interaction: click the first visible button (else the first canvas), press ArrowRight, ArrowUp, Space, Enter (150 ms page time each), type 42 into the first visible text/number input, 1 s page time, `interact.png`. Screenshots with `animations="disabled"`. `renders` = loaded and no uncaught error up to `load.png`.
- *Embeddings:* screenshots with `google/siglip2-base-patch16-384` (snapshot f775b65a), code with `Qwen/Qwen3-Embedding-0.6B` (snapshot 97b0c614, no instruction prefix, max 16,384 tokens); both L2-normalised, MPS.
- *Screen embedding* of an app = normalised mean of its two screenshot embeddings. *Outcome distance* (spec §5.4) = 1 if exactly one of the two apps renders, else screen cosine distance clipped to [0, 1]. *Ensemble spread / rarity* (frozen plan) = mean pairwise / mean-to-others screen cosine distance **without** the render rule; the render-rule variant is reported alongside.
- **Deviation, labelled:** prompt adherence for tight prompts (H4 secondary, spec §5.5 "hand-written checklist") is taken from the Claude Chrome checklist (D-030) instead of a second, automatic checklist. A Claude-derived measure thus enters a pre-registered secondary test; it is reported as "adherence (Claude checklist)" and never pooled with human ratings.

**Reason:** Screens must be comparable between grid apps and fork rollouts, and same-vs-same rollout distances (the noise floor of fork importance) must come from the model, not the renderer. Check on 30 random grid apps rendered twice: 29 byte-identical screenshot pairs, one pair differed once (CSS animation timing race; four further renders identical). Without the paused clock and disabled animations, animated apps differed on every run.

**Alternatives:** Real time and real randomness (noisy screens); a hand-written automatic adherence checklist per tight prompt (duplicates the Claude checklist, which already exists for all 180 tight apps); CLIP ViT-B/32 (older, 224 px).
