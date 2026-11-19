# Methods

All information quantities are in bits (log base 2). Let p be the full next-token distribution at step t and x_t the sampled token.

- **Entropy H** = −Σ p log₂ p
- **Surprisal S** = −log₂ p(x_t)
- **Varentropy** = Var_p[−log₂ p] = Σ p (log₂ p)² − H²
- **Margin** = log₂ p(top1) − log₂ p(top2)

Implementation notes are filled in Task 2.
