# Data formats

Run directory: `<ENTROPYTRAIL_DATA>/<condition_id>/<prompt_id>/<sample:03d>/`
(`ENTROPYTRAIL_DATA` defaults to `/Volumes/Data/entropytrail`). One directory is one
generation unit. It is written atomically: all files go to `<sample:03d>.partial/`, which
is renamed to `<sample:03d>/` only when complete. A `.partial` directory is never valid
data and is deleted by `clean_partials`; a unit is done iff `meta.json` has `status: "done"`.

All information quantities are in **bits** (log base 2). Full logits are never stored.

## output.txt

Raw model output, UTF-8, exactly as decoded (may include markdown fences or prose).

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

JSON (indent 2, UTF-8). Keys supplied by the caller (condition, prompt id, sample, seed,
...) are kept as-is; storage adds/overrides:

| key | type | unit | definition |
|---|---|---|---|
| `n_tokens` | int | tokens | number of generated tokens (rows of `tokens.parquet`) |
| `finish_reason` | string | | `"eos"` or `"max_tokens"` (truncated) |
| `seconds` | float | s | wall-clock generation time |
| `tokens_per_s` | float | tokens/s | `n_tokens / seconds`, 0.0 if `seconds == 0` |
| `html_extracted` | bool | | whether `app.html` was written |
| `status` | string | | always `"done"` in a completed unit |
