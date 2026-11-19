# Methods

All information quantities are in bits (log base 2). Let p be the full next-token distribution at step t and x_t the sampled token.

- **Entropy H** = −Σ p log₂ p
- **Surprisal S** = −log₂ p(x_t)
- **Varentropy** = Var_p[−log₂ p] = Σ p (log₂ p)² − H²
- **Margin** = log₂ p(top1) − log₂ p(top2)

## Implementation notes (`src/entropytrail/stats.py`)

- Logits and log-softmax are computed in float32 from a float32 copy of `lm_head` (condition field `logits_precision: float32`); the transformer body stays bf16.
- Input is the natural-log probability vector from `generate_step` (log-softmax of the logits). With the float32 head it is already float32 (the cast to float32 in `stats.py` only matters for a condition with `logits_precision: native`, where log-probs are in the model dtype bf16); it is converted to bits by dividing by ln 2, and all four quantities are computed on device in float32.
- Entries with p = 0 (log-prob = -inf) contribute exactly 0 to H and to the second moment (`where(p > 0, ..., 0)`), so no NaNs arise. Varentropy is clamped at 0 against rounding.
- Surprisal S uses the log-prob of the token actually emitted (or, in teacher forcing, the forced token).
- Margin uses the two largest log2 probabilities (top-1 minus top-2); 0 if the vocabulary has a single entry.
- Top-k: k = min(20, V), found with `argpartition` and then sorted descending; stored as token ids (int32) and log2 probabilities (float32).

## Precision

Recorded values are the exact sampling distribution: the transformer body runs in bf16, the head (float32 copy of `lm_head`) and the log-softmax in float32. A chunked teacher-forced recomputation (chunk > 1) differs from the recorded values because the bf16 body kernels round differently for different chunk sizes; measured on 1500 tokens of the smoke app (chunk 512): p99 |dH| 0.044, p99 |dS| 0.083, max |dH| 0.095, max |dS| 0.254 bits. Compare against recorded values only by recomputing token by token (chunk = 1), which reproduces them exactly (decision D-021).
