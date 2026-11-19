# Decisions

Format: `## D-NNN · YYYY-MM-DD · title` with Decision / Reason / Alternatives. Every choice made while building is recorded here.

## D-001 · 2026-11-19 · Approach C

**Decision:** Own measurement pipeline on a local model, generation and token-level recording in one pass, followed by fork experiments.

**Reason:** Gives full control over the distribution at each step and keeps the study reproducible.

**Alternatives:** A: hosted API with logprobs only; B: post-hoc scoring of existing apps.

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
