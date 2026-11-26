# Theory: generation as a process on a tree, apps as its projection

This note is a reading frame for the measurements the project already makes (H1–H5, the temperature sweep, D-041, D-042). It adds no hypotheses and no experiments. It grew out of Robert's notes "Entropie-Trail – Notizen zur Theorie und zu nächsten Messungen" (2026-11-26). Items deliberately left for later are listed at the end.

## 1. State and dynamics

The **state** at step t is the prefix x_t: the prompt (prefill) followed by the tokens generated so far. One step appends one token:

  x_{t+1} = x_t ⊕ y_{t+1},  y_{t+1} ~ p_θ(· | x_t)

The network computes the next-token distribution p_θ deterministically from the prefix (the drift). Sampling at temperature T is the only source of randomness (the noise). Its local strength is the entropy of the distribution, H_t. Everything inside the network (KV cache, the recurrent state of Qwen3.5's linear-attention layers) is a deterministic function of the prefix. So the prefix is a complete state, and the process is Markov in it.

The state space is a **tree**: every prefix has about 250,000 children and exactly one parent. The end token is an absorbing transition, a random stopping time. The **final states** are the complete sequences.

**What the code is.** The model does not separate "token chain" and "HTML": everything after the prefill is one chain. In the data it is practically the code. At T = 1.0, 401 of 440 outputs are a fenced ```` ```html ```` block and the rest start directly with the document. Text outside the document is at most 884 characters before it (median 8) and at most 4 after it. Cutting the document out is post-processing (`src/entropytrail/extract.py`). The only part of the chain that is not code is the stop decision.

## 2. Why the reachable set says nothing, and the distribution does

In control terms the token choice is the control. Softmax gives every token positive probability, so **every** finite sequence is reachable at any T > 0. The reachable set is everything; the model decides only how likely each sequence is. This mirrors an SDE with non-degenerate noise, whose law has full support (Stroock–Varadhan). At T → 0 (greedy decoding) the process is deterministic, and the region collapses to one trajectory.

What carries information is the **distribution over final states**. A set notion comes back as its high-mass part: the smallest set holding 1 − δ of the probability (or the states with probability ≥ ε). We call this the **attainable region**. It shrinks as T falls. The 20 samples per prompt are draws from it.

## 3. The forward equation is trivial on the tree

For an SDE the density obeys the Fokker–Planck equation. It is non-trivial because diffusion moves mass into neighbouring states and paths merge. Here there is no small step and no neighbourhood: each step is a jump to one of the children. The analogue is the master equation (forward Kolmogorov equation):

  P_{t+1}(x ⊕ y) = P_t(x) · p_θ(y | x)

On a tree probability only splits, never merges. The solution is the product along the path, P(x) = ∏_s p_θ(y_s | x_{s−1}). Entropy grows by exactly the expected per-token entropy at each step (chain rule):

  H(Y) = E[ Σ_t H_t ]

So the mean of ΣH over the samples of a prompt estimates the entropy of its whole sequence distribution. At T ≠ 1 this needs the entropy of the sampling distribution, H_T (rescoring, D-042), because the recorder stores the untempered H. Each individual sequence has probability 2^(−ΣS), hundreds of bits for a typical app. Every sampled sequence is therefore practically unique, and "which sequences were reached" carries no information.

## 4. Projection onto apps: where the dynamics become non-trivial

Many sequences give the same app. A hierarchy of many-to-one maps f, from fine to coarse:

| level | same class if … | measured by |
|---|---|---|
| token sequence | identical tokens | `tokens.parquet` |
| text | identical string (different tokenisations merge) | `output.txt` |
| HTML | identical extracted document | `app.html` |
| normalised code | identical after whitespace, comments, names | (later) |
| rendering | same screen | screenshot embeddings, ensemble spread |
| behaviour | same reactions to the same interactions | Claude ratings (functionality, checklist) |
| idea | same core idea | idea classes (D-042) |

The distribution of apps is the pushforward f#P: P(f = c) = Σ_{y: f(y)=c} P(y). The **attainable region at a level** is the high-mass set of f#P. Mass concentrates only on classes, never on single sequences.

Along the hierarchy entropy can only fall, H(f(Y)) ≤ H(Y). The difference H(Y) − H(f(Y)) = H(Y | f(Y)) is the entropy spent on things that do not matter at that level: spelling, formatting, naming, equivalent ways of writing the same thing. This **entropy cascade** is the quantitative form of the quotient picture. The expectation is thousands of bits at the token and HTML levels and a few bits at the idea level.

The projected process is **not Markov**: the class of a half-written app does not determine how it continues. This is the Mori–Zwanzig situation: projecting microscopic dynamics (tokens) onto coarse variables (app classes) yields drift, a memory term and noise. Paths that never merge on the tree do merge in the projection. Everything non-trivial happens there.

## 5. Lock-in, defined on the projection

Let π_t(c) = P(f(Y) = c | x_t) be the distribution of the final class given the prefix. It starts at the prompt's app distribution and ends as a point mass on the generated app. In between it is a martingale (Doob): in expectation it does not drift, it only narrows. **Lock-in is the time at which π_t becomes concentrated** (e.g. H(π_t) small): from then on, the trajectory can only end in few classes.

The per-token entropy H_t is only a proxy for this. A token can have high entropy and leave π_t unchanged (`button` vs `btn`), or low entropy and move it a lot. Falling H_t late in an app can also be closing boilerplate (`</script></html>`) rather than lock-in. D-042 therefore reports entropy over position per section, and the JS slope with and without its last 10 %.

## 6. What the existing measurements are in this frame

| measurement | in this frame |
|---|---|
| H, S per token (H1) | local noise strength along the path; ΣH estimates H(Y) |
| forks: share of tokens with H ≥ 2 bits | where the tree branches with non-negligible mass |
| fork experiments, importance (H3) | local response of π_t to a single forced token; the same-token re-samples are a small Monte Carlo estimate of π_t at that prefix. H3 asks whether H_t predicts how far π_t moves |
| ensemble spread (H2, C2) | width of f#P at the rendering level |
| renders, functionality (C1, H4) | mass of f#P on working apps |
| novelty: calibrated (C3), cross-temperature (D-041) | rarity of a class within f#P (one temperature) or within the pooled distribution |
| semantic entropy (D-042) | H(f(Y)) at the idea level |
| temperature sweep | changing the noise strength; lower T concentrates P and f#P |
| stop probabilities, length distribution (D-042) | the stopping time of the absorbed process |

## 7. Deliberately deferred

To be picked up after H3 and the T = 0.5 sweep are reported, in separate discussions: estimating π_t along whole trajectories (lock-in curves); level-2 code normalisation; a formal SDE or continuum description (needs a scaling argument, which the jump process lacks); the entropy-cascade figure across all levels (it becomes possible once rescoring and D-042 are done); TabPFN as a surrogate model for CFD and other topics from the notes outside this project.
