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
