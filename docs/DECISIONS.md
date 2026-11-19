# Decisions

Format: `## D-NNN · YYYY-MM-DD · title` with Decision / Reason / Alternatives. Every choice made while building is recorded here.

## D-001 · 2026-11-19 · Approach C

**Decision:** Approach C: A first, then B on the same recording core. A is one-shot generation with many samples per prompt, full distributions logged and fork rollouts; B is a crafting loop (draft, render/feedback, revise) with all rounds logged.

**Reason:** Find out whether a pattern exists before building the loop; one shared logger.

**Alternatives:** A alone (cleanest view of the landscape and reachability, but no iterative crafting); B alone (real crafting process, but much more machinery and feedback noise mixes into the model's own decisions); the original plan, observer-only measurement of externally produced apps, rejected as core because it cannot show the generator's landscape.

## D-002 · 2026-11-19 · Primary model Qwen3.5-9B in bf16

**Decision:** Qwen/Qwen3.5-9B at bf16 is the primary condition.

**Reason:** Fits the Mac Studio (32 GB) without quantisation noise in the distributions.

**Alternatives:** Quantised variants; larger models.

## D-003 · 2026-11-19 · Local only, no Ollama for measurement

**Decision:** All measurement runs use MLX directly on the local machine.

**Reason:** Ollama does not expose full distributions; one runtime per condition keeps conditions comparable.

**Alternatives:** Ollama with logprobs.

## D-004 · 2026-11-19 · Prompt grid 3x2 plus free prompt

**Decision:** Prompts form a 3x2 grid plus one free prompt (spec 5.1).

**Reason:** Enough variation for contrasts while staying computable.

**Alternatives:** Larger grid; a single prompt.

## D-005 · 2026-11-19 · 20 samples per prompt, T=1.0, no truncation

**Decision:** 20 samples per prompt, temperature 1.0, no top-p/top-k truncation, max 16,000 tokens.

**Reason:** Samples the model's true distribution, which is what the entropy measures describe.

**Alternatives:** Lower temperature; nucleus sampling.

## D-006 · 2026-11-19 · Thinking mode off (D1 proposal)

**Decision:** Thinking is disabled via enable_thinking=False in the chat template.

**Reason:** Keeps the recorded trail on the artifact itself and the output length bounded.

**Alternatives:** Thinking on as a separate condition later.

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

**Decision:** The Mac Studio produces the main series; the desktop PC is a separate condition.

**Reason:** One condition equals one machine, runtime and precision.

**Alternatives:** Pooling machines.

## D-011 · 2026-11-19 · Run window 23:00-06:00

**Decision:** Runs happen only between 23:00 and 06:00; a unit interrupted by the deadline is discarded.

**Reason:** The machine is otherwise in use.

**Alternatives:** Continuous runs.

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

**Reason:** Native Apple Silicon speed; pinning keeps conditions reproducible (decided in the foundation plan).

**Alternatives:** PyTorch MPS; llama.cpp.

## D-017 · 2026-11-19 · Validation tolerance exceeded on the 9B model; thresholds unchanged

**Decision:** The validation run on the real model (smoke-tool-01, 1500 tokens) gave p99 |dH| 0.0823 bits, p99 |dS| 0.1803 bits, max |dH| 0.2912, max |dS| 0.3607, same_tokens true, i.e. above the 0.01-bit threshold. The threshold and the code were not changed; the result is reported to Robert for a decision (likely cause to investigate: bf16 numerical differences between incremental decoding with cache and chunked full-sequence teacher forcing).

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
