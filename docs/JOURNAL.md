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

## 2026-11-19 (fix round 2: validation criterion)

Robert decided the criterion (D-021): validation passes iff chunk=1 teacher forcing reproduces recorded H and S within 1e-5 bits max and same_tokens; chunked recomputation is informational (`chunked_noise`). Re-run on the 9B, 1500 tokens, exit 0: chunk=1 max dH/dS 0.0, ok true, same_tokens true; chunked_noise (chunk 512) p99 dH 0.0439, p99 dS 0.0829, max dH 0.0947, max dS 0.2543 bits; generate 15.6 tok/s. Slow tests pass after the head swap.

## 2026-11-19 (final review fixes)

Final review fixes of plan 1 (D-022 to D-026): condition manifest `condition.json` and resume identity guard (prompt and system-prompt hashes, `prompt_ids` in meta.json); mlx-lm version checked at model load; `et generate` / `et run-window` refuse a dirty git tree without `--allow-dirty`, and `validation.json` carries provenance and the weights fingerprint; runlog entry written also on crash (`error`, `git_commit`, `weights_fingerprint`, `stale_moved`), non-done final directories moved aside and redone; night script restarts Ollama only if it stopped it and runs `uv run --frozen --offline`; DATA.md documents every file and key; DECISIONS entries D-002/3/4/6/10/11/16/20 aligned with the spec. Condition fields unchanged, condition_id stays `qwen35-9b-bf16-mlx-1195d88b` (now pinned in a test). The existing validation and smoke data predate these keys; validate/smoke are to be re-run at a clean HEAD.

## 2026-11-19 (clean re-run at 82adefd)

Re-ran on the 9B at a clean tree (git_dirty false, commit 82adefd). `et validate --tokens 1500`: exit 0; chunk=1 max dH/dS 0.0, same_tokens true; chunked_noise (chunk 512) p99 dH 0.0439, p99 dS 0.0829 bits — identical to the earlier run; generate 14.9 tok/s. `condition.json` created for `qwen35-9b-bf16-mlx-1195d88b` (weights_fingerprint f03e7e8f…, mlx 0.32.4, mlx-lm 0.32.2). Smoke unit `smoke-tool-01/000` regenerated: 4176 tokens, eos, 17.0 tok/s, html_extracted true — same length as the earlier dirty-tree run with the same seed, i.e. reproducible across the code changes. The earlier unit was moved to `smoke-tool-01/000.superseded-dirty-6df192d` (not deleted). Visual check of app.html: done by Robert, looks OK.

## 2026-11-20 (first night run)

First night of `de.entropytrail.night` (queue: smoke-tool-01, 20 samples). Started 23:00; the Python process blocked ~50 min in `open()` on a macOS privacy prompt for access to the external volumes (launchd-started processes do not inherit the interactive grant). Robert clicked Allow on the Studio; the run continued 23:49–00:30: 19 units done, 1 skipped (000), 0 aborted, no error, commit a30ba4c, Ollama restarted (0.40.14). Sample 001: 2743 tokens, 15.6 tok/s. Consequence: watchdog added to the night script (D-027).

## 2026-11-20 (WP2 frozen)

Prompt grid v1 (21 prompts) and the analysis plan frozen after Robert's approval (D-028). Night queue now runs the full grid, 20 samples per prompt (420 units); expected ~3–4 nights at ~16 tok/s.
