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
