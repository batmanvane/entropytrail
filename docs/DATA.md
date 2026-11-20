# Data formats

Run directory: `<ENTROPYTRAIL_DATA>/<condition_id>/<prompt_id>/<sample:03d>/`
(`ENTROPYTRAIL_DATA` defaults to `/Volumes/Data/entropytrail`). One directory is one
generation unit. It is written atomically: all files go to `<sample:03d>.partial/`, which
is renamed to `<sample:03d>/` only when complete. A `.partial` directory is never valid
data and is deleted by `clean_partials`; a unit is done iff `meta.json` has `status: "done"`.
A final directory without `status: "done"` is stale: the next run renames it to
`<sample:03d>.stale-<YYYYmmdd-HHMMSS>/` and redoes the unit (logged in `runlog.jsonl`).

Files of a unit: `output.txt`, `app.html` (if extracted), `tokens.parquet`, `meta.json`.
Per condition: `<condition_id>/condition.json`. Per data root: `runlog.jsonl`,
`_validation/`, `_bench/`.

All information quantities are in **bits** (log base 2). Full logits are never stored.

## output.txt

Raw model output, UTF-8, exactly as decoded (may include markdown fences or prose); the
concatenation of `tokens.parquet: text` in order (the EOS token contributes no text).

## app.html

The HTML document extracted from `output.txt` by `extract_html` (fences/prose stripped).
Present only if extraction succeeded (`meta.json: html_extracted == true`).

## tokens.parquet

Logits and log-softmax are computed in float32 from a float32 copy of `lm_head` (condition field `logits_precision`).

One row per generated token, in generation order.

| column | type | unit | definition |
|---|---|---|---|
| `pos` | int32 | index | 0-based position in the generated sequence |
| `token_id` | int32 | id | sampled token id |
| `text` | string | chars | decoded text of this token |
| `char_offset` | int32 | characters | start offset in `output.txt`; equals `len("".join(texts[:pos]))` |
| `H` | float32 | bits | Shannon entropy of the full next-token distribution |
| `S` | float32 | bits | surprisal of the sampled token, `-log2 p(token)` |
| `varentropy` | float32 | bits^2 | variance of surprisal under the distribution, `E[(-log2 p)^2] - H^2` |
| `margin` | float32 | bits | `log2 p(top1) - log2 p(top2)` (0 if vocab < 2) |
| `topk_ids` | list<int32> | id | ids of the top k=20 tokens, descending probability |
| `topk_logp2` | list<float32> | bits | `log2` probability of each `topk_ids` entry (same order) |

## meta.json

JSON (indent 2, UTF-8), one per unit. A night run (`et generate` / `et run-window`,
`runner.run_units`) writes these keys:

Provenance (`model.provenance()`, at the start of the run):

| key | type | definition |
|---|---|---|
| `git_commit` | string | `git rev-parse HEAD` of the repo, `"unknown"` if git is unavailable |
| `git_dirty` | bool or null | uncommitted changes in the working tree; null if unknown. `et generate` / `et run-window` refuse to run when true unless `--allow-dirty` is given |
| `mlx` | string | installed `mlx` version |
| `mlx_lm` | string | installed `mlx-lm` version (must equal the condition's `runtime_version`, checked at model load) |
| `python` | string | Python version |
| `macos` | string | macOS version (`platform.mac_ver()`) |
| `hostname` | string | host name of the machine |

Identity of the unit (runner):

| key | type | definition |
|---|---|---|
| `condition` | object | all fields of the condition (`dataclasses.asdict(Condition)`) |
| `condition_id` | string | `<name>-<first 8 hex of sha256 of the canonical JSON of condition>` |
| `prompt` | object | the prompt: `id`, `domain`, `openness`, `text` |
| `prompt_set_version` | string | `version` of the prompt set used (`v1` = main grid, `v0-smoke` = recorder tests) |
| `system_prompt_version` | string | key of the system prompt used (from the condition) |
| `system_prompt_sha256` | string | sha256 (hex) of the system prompt text (UTF-8) |
| `prompt_text_sha256` | string | sha256 (hex) of the user prompt text (UTF-8); checked against the current prompt when a done unit is skipped on resume |
| `prompt_ids` | list<int> | token ids fed to the model (chat template applied with system and user prompt, `enable_thinking` from the condition); covers tokenizer and chat template |
| `sample` | int | sample index within the prompt (0-based) |
| `seed` | int | `derive_seed(condition_id, prompt_id, sample)`: first 4 bytes (big-endian) of sha256 of `"<condition_id>|<prompt_id>|<sample>"` |
| `weights_fingerprint` | string | see below |
| `started_at`, `finished_at` | string | local ISO 8601 timestamps of the generation of this unit |

Added by storage (`write_run`):

| key | type | unit | definition |
|---|---|---|---|
| `n_tokens` | int | tokens | number of generated tokens (rows of `tokens.parquet`) |
| `finish_reason` | string | | `"eos"` or `"max_tokens"` (truncated) |
| `seconds` | float | s | wall-clock generation time |
| `tokens_per_s` | float | tokens/s | `n_tokens / seconds`, 0.0 if `seconds == 0` |
| `html_extracted` | bool | | whether `app.html` was written |
| `status` | string | | always `"done"` in a completed unit |

### weights_fingerprint

sha256 (hex) over the sorted blob names of the `*.safetensors` files of the local HF
snapshot that `refs/main` points to, joined by `\n`. In the HF cache a blob name is the
sha256 of the file content, so the fingerprint identifies the exact weight files. It does
not cover the tokenizer, the chat template or `config.json`; `prompt_ids` in each
`meta.json` record the tokenization actually used.

## condition.json

`<ENTROPYTRAIL_DATA>/<condition_id>/condition.json`, written once by the first run of a
condition. Every later run compares all keys except `created_at` and aborts (no unit is
run, the runlog entry carries the error) if any differs. A done unit is additionally
checked on resume: its `prompt_text_sha256` (or, for units written before this key
existed, the sha256 of its stored `prompt.text`) must equal that of the current prompt.

| key | type | definition |
|---|---|---|
| `condition` | object | all condition fields |
| `condition_id` | string | as in meta.json |
| `weights_fingerprint` | string | as in meta.json |
| `system_prompt_version` | string | key of the system prompt |
| `system_prompt_sha256` | string | sha256 of the system prompt text |
| `mlx`, `mlx_lm` | string | installed versions |
| `created_at` | string | local ISO 8601 time of the run that created the file (not compared) |

## runlog.jsonl

`<ENTROPYTRAIL_DATA>/runlog.jsonl`, one JSON line appended per `run_units` call
(also when it fails; the exception is then re-raised).

| key | type | definition |
|---|---|---|
| `started`, `ended` | string | local ISO 8601 times of the call |
| `condition_id` | string | condition of the run |
| `done` | int | units generated and written |
| `skipped` | int | units already done (not recomputed) |
| `aborted` | int | 1 if a unit was interrupted by the deadline and discarded, else 0 |
| `error` | string or null | null on success, else `"<ExceptionType>: <message>"` |
| `git_commit` | string | as in meta.json |
| `weights_fingerprint` | string | as in meta.json |
| `stale_moved` | list<string> | final unit directories that existed without `status: "done"` and were renamed to `<sample:03d>.stale-<YYYYmmdd-HHMMSS>` (paths relative to the data root) before the unit was redone |

## _validation/&lt;condition_id&gt;/&lt;YYYYmmdd-HHMMSS&gt;/validation.json

Written by `et validate` (decision D-021). Generates the first prompt of the prompt set
(sample-0 seed, `--tokens` tokens), scores it by teacher forcing and regenerates it.

| key | type | definition |
|---|---|---|
| `bookkeeping_chunk1` | object | teacher forcing token by token vs recorded values: `p99_abs_dH`, `p99_abs_dS`, `max_abs_dH`, `max_abs_dS` (bits), `ok` = max of both <= 1e-5 bits |
| `chunked_noise` | object | same differences for chunked teacher forcing (`chunk`, default 512), informational only: `p99_abs_dH`, `p99_abs_dS`, `max_abs_dH`, `max_abs_dS`, `chunk` |
| `ok` | bool | `bookkeeping_chunk1.ok` |
| `same_tokens` | bool | regeneration with the same seed gave the same token ids |
| `condition_id`, `prompt_id`, `seed`, `n_tokens` | | the validated generation |
| `generate_tokens_per_s` | float | tokens/s of the first generation |
| `score_tokens_per_s` | float | `2 * n_tokens / scoring time` (both teacher-forcing passes) |
| `weights_fingerprint` | string | as in meta.json |
| `git_commit`, `git_dirty`, `mlx`, `mlx_lm`, `python`, `macos`, `hostname` | | provenance, as in meta.json |

Exit code 0 iff `ok` and `same_tokens`. Files written before 2026-11-19 (final review
fixes) lack the provenance keys and `weights_fingerprint`.

## _bench/&lt;YYYYmmdd-HHMMSS&gt;.json

Written by `et bench` (fixed prompt, system prompt "You are a helpful assistant.",
seed 0, `--tokens` tokens, single sequence).

| key | type | unit | definition |
|---|---|---|---|
| `condition_id` | string | | condition benchmarked |
| `prompt_tokens` | int | tokens | prompt length |
| `tokens` | int | tokens | generated tokens |
| `prefill_s` | float | s | time to the first token |
| `prefill_tokens_per_s` | float | tokens/s | `prompt_tokens / prefill_s` |
| `decode_s` | float | s | time from first to last token |
| `decode_tokens_per_s` | float | tokens/s | `(tokens - 1) / decode_s` |
