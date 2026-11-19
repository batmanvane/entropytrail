# Journal

Format: `## YYYY-MM-DD` entries.

## 2026-11-19

Project set up: repository skeleton, environment (uv, Python 3.12, mlx-lm 0.32.2) and documentation skeleton.

## 2026-11-19 (validation and benchmark of the 9B recorder)

First run of the recorder on the real Qwen3.5-9B (bf16, MLX, mlx-lm 0.32.2) on the Mac Studio, with Ollama idle (no model loaded).

- Slow tests (`pytest -m slow`): 3 passed.
- `et bench`, 1000 tokens, 38 prompt tokens: first run was cold (weights read from disk): prefill 156.4 s, decode 12.7 tok/s (not representative). Warm rerun: prefill 3.2 s for 38 tokens (11.8 tok/s, dominated by warm-up; the prompt is too short to measure prefill rate), decode 21.1 tok/s.
- `et validate`, smoke-tool-01, 1500 tokens: **FAILED the threshold.** p99 |dH| = 0.0823 bits, p99 |dS| = 0.1803 bits, max |dH| = 0.2912, max |dS| = 0.3607; same_tokens = true; generate 19.5 tok/s, teacher-forced scoring 36.3 tok/s. Threshold (p99 <= 0.01 bits) exceeded by ~8-18x. No threshold or code was changed; see D-017.
- Smoke app (`et generate`, smoke-tool-01, sample 0, seed 1893739985): 2,731 tokens, finish_reason eos, 136 s (20.1 tok/s), html_extracted true; app.html 10 kB, starts with `<!DOCTYPE html>`, ends with `</html>`, one `<script>`, no external resources (static check only; visual check pending). tokens.parquet schema and meta.json keys match docs/DATA.md. Data root `/Volumes/Data/entropytrail` had to be created.
- Spec section 8 time table replaced by an estimate from these numbers (main grid ~16 h, ~3 nights; forks 42 apps ~51 h, ~8 nights).

## 2026-11-19 (fix round 1: float32 logits head)

- Failure: validation p99 dH 0.0823 / dS 0.1803 bits (see above). Controller probe on the first 800 tokens: teacher forcing with chunk=1 reproduces recorded H and S exactly (diff 0.0), so bookkeeping is correct. With chunk 16/512 the median diff is 2e-6 bits but p99 dH 0.05-0.09 and max dS is exactly 0.180 / 0.361 bits = one bf16 ulp of log-probs at |logp| 8-32 nats. Cause: qwen3_5 `lm_head` outputs bf16 logits and the log-softmax runs in bf16.
- Fix: float32 copy of `lm_head` (`logits_precision: float32` in the condition, new condition_id `qwen35-9b-bf16-mlx-1195d88b`); `et validate` reports a chunk=1 and a chunk=512 block (D-020).
- New numbers: bench (warm) decode 18.9 tok/s. Validate, 1500 tokens: chunk=1 max dH/dS 0.0 (ok); chunk=512 p99 dH 0.0439, p99 dS 0.0829, max dH 0.0947, max dS 0.2543 bits (**still above 0.01, not ok**); same_tokens true; generate 17.0 tok/s. Thresholds not loosened; no further fix tried (D-017).
- Smoke app under the new condition: 4,176 tokens, eos, 241 s (17.3 tok/s), html_extracted true, static check OK, no external resources.
- Spec section 8 table updated (main grid ~28 h / ~4 nights, forks ~90 h / ~13 nights).
- The data of the old condition `qwen35-9b-bf16-mlx-6466454b` under /Volumes/Data/entropytrail is superseded and left in place.
