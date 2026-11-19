# Methods

All information quantities are in bits (log base 2). Let p be the full next-token distribution at step t and x_t the sampled token.

- **Entropy H** = −Σ p log₂ p
- **Surprisal S** = −log₂ p(x_t)
- **Varentropy** = Var_p[−log₂ p] = Σ p (log₂ p)² − H²
- **Margin** = log₂ p(top1) − log₂ p(top2)

## Implementation notes (`src/entropytrail/stats.py`)

- Input is the natural-log probability vector from `generate_step` (log-softmax of the logits). It is cast to float32 (the model runs in bf16), converted to bits by dividing by ln 2, and all four quantities are computed on device in float32.
- Entries with p = 0 (log-prob = -inf) contribute exactly 0 to H and to the second moment (`where(p > 0, ..., 0)`), so no NaNs arise. Varentropy is clamped at 0 against rounding.
- Surprisal S uses the log-prob of the token actually emitted (or, in teacher forcing, the forced token).
- Margin uses the two largest log2 probabilities (top-1 minus top-2); 0 if the vocabulary has a single entry.
- Top-k: k = min(20, V), found with `argpartition` and then sorted descending; stored as token ids (int32) and log2 probabilities (float32).
