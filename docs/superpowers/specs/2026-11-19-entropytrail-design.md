# Entropy Trail – Design

Status: draft for approval · 2026-11-19 · Robert (with Claude)

Supersedes the scope of `Entropy Trail – Projektbeschreibung.md` (kept as the origin document). trez63.com is only the inspiration (the kind of small single-file HTML app); no trez63 apps are built, scored or used as data.

## 1. Intent

Understand **how decisions emerge** inside a language model while it builds something.

Working picture (Robert's): an LLM holds an abstract probability landscape over everything it could produce next. Every emitted token reshapes that landscape – some paths close, others open. "What can be generated" is what is still *reachable* from the current position. The final output is one path out of many, in the spirit of a path integral.

Questions:

1. Is there a **pattern** in how the landscape reshapes during the construction of a non-trivial artifact?
2. Is entropy (or a related quantity) a good indicator of whether an output will be **useful**?
3. Is it an indicator of whether an output will be **new / innovative**?

Small single-file HTML apps (as on trez63.com) are the test object: richer than single sentences, clearly structured, with identifiable decision points, and checkable (does it run, what does it look like).

This is a basic-research study. Good documentation is a first-class goal: every run, decision and result must be reproducible and traceable from the repo.

## 2. Goals, non-goals, anti-goals

**Goals**
- Record the full next-token landscape of a local model while it builds HTML apps, for many runs per prompt.
- Measure reachability directly by going back to forks and rolling out alternatives.
- Collect independent judgments of usefulness and novelty (human ratings plus automatic checks) to test whether landscape measures predict them.
- Phase B: extend to an iterative crafting loop (draft → feedback → revision) on the same recorder.
- Complete documentation: design, decision log, data dictionary, run journal, analysis notebooks.

**Non-goals**
- Measuring the internal landscape of closed models (Claude, Gemini, ChatGPT, …). Their apps can only be *observed* by a local model, which says nothing about their own landscape.
- Using or scoring the trez63.com apps; trez63 is only the inspiration.
- Building a polished public product or a daily publishing pipeline.
- Frameworks, build tools or multi-file apps. Output is one self-contained HTML file, no external libraries.

**Anti-goals (would make this a failure even if it runs)**
- Numbers that cannot be traced back to an exact model, weights hash, prompt, seed and code version.
- Changing the model, quantization or sampling settings mid-study without versioning it as a new condition.
- Treating observer-model entropy on foreign apps as if it were the generator's landscape.
- A pile of raw data with no documented definitions.

## 3. Approaches considered

| | Approach | Trade-off |
|---|---|---|
| A | One-shot generation, many samples per prompt, full distributions logged, fork rollouts | Cleanest view of the landscape and reachability; no iterative crafting |
| B | Crafting loop: draft → render/feedback → revise, all rounds logged | Real crafting process; much more machinery, feedback noise mixes into the model's own decisions |
| **C (chosen)** | **A first, then B on the same recording core** | Find out if a pattern exists before building the loop; one shared logger |

Rejected as core: observer-only measurement of externally produced apps (original plan) – it cannot show the generator's landscape.

## 4. Constraints and environment

- Machine: Mac Studio M2 Max, 32 GB unified memory. Data on `/Volumes/Data` (~690 GB free), models on `/Volumes/Models` (`HF_HOME=/Volumes/Models/huggingface`).
- No deadline. Compute budget is open; scale is set by measured run times, not by a date.
- During runs the Ollama service on the Studio (`com.robert.ollama`) is stopped to free memory for the 9B model (Robert approved, 2026-11-19); it is restarted afterwards.
- Local only. Ollama is not usable for measurement (no full distributions, no teacher forcing); use MLX or PyTorch directly.
- **Primary model: Qwen3.5-9B** at full precision (bf16), frozen for the whole study. Weights are in the HF cache (4 safetensors files, 19.3 GB; checked 2026-11-19). Runtime: mlx-lm 0.32.2, which supports the `qwen3_5` architecture (hybrid: linear-attention layers with a full-attention layer every 4th). Facts from its config: architecture `Qwen3_5ForConditionalGeneration` (multimodal – usable for screenshot feedback in phase B), vocabulary 248,320, hidden size 4096, 32 layers, context 262,144.
- **Comparison model (later): Qwen3.6-35B-A3B**, MLX 4-bit (already on disk). Quantization distorts probabilities; it is a second condition, never mixed with the 9B data.
- Raw data lives outside the repo under `/Volumes/Data/entropytrail/`. The repo holds code, docs, prompt set, manifests and derived summaries.

## 5. Experimental design (phase A)

### 5.1 Prompt grid

3 domains × 2 openness levels + a free cell. About 3 prompts per cell → ~21 prompts.

| | Tight brief | Loose brief | Open |
|---|---|---|---|
| Education | e.g. "Quiz with 10 questions on fractions, score at the end" | e.g. "An app that helps students understand fractions" | |
| Game | e.g. "Snake on a 20×20 grid, arrow keys, score" | e.g. "A simple game for a break" | |
| Tool | e.g. "Unit converter cm/inch/feet" | e.g. "A small everyday helper" | |
| Free | | | "Build an HTML app of your choice" (+ 2 variants) |

All prompts share one fixed system prompt stating the format constraints: one self-contained HTML file, inline CSS/JS, no external resources, output only the file. Prompts are versioned in `prompts/prompts.yaml` with stable IDs (e.g. `edu-tight-01`).

Openness level is an experimental variable: hypothesis is that it moves the forks from details (tight) to the idea level (loose/free).

### 5.2 Generation runs

- 20 samples per prompt → ~420 apps.
- Sampling: temperature 1.0, **no top-p/top-k truncation**, so samples are drawn from the true landscape. Seeds recorded per sample.
- Max output length: 16k tokens; truncated runs are kept and flagged.
- Reasoning ("thinking") mode: see open decision D1.

### 5.3 What is recorded per generated token

| Field | Definition |
|---|---|
| `token_id`, `text`, `char_offset` | the sampled token and where it sits in the output |
| `H` | entropy of the full next-token distribution, bits: −Σ p log₂ p |
| `S` | surprisal of the sampled token, bits: −log₂ p(x_t) |
| `varentropy` | variance of −log₂ p under p |
| `margin` | log₂ p(top1) − log₂ p(top2) |
| `topk` | top-20 token ids with log-probs |
| `section` | HTML / CSS / JS / comment / text-content / other (derived after generation) |

Full logits are **not** stored (248k floats × 10k tokens ≈ 10 GB per app). H and varentropy are computed from the full distribution at generation time.

Hidden states: for a subset (see D3), residual-stream vectors at a few layers, float16, for the representation-trajectory analysis.

### 5.4 Fork experiments (reachability)

For each app:
1. Select fork positions: top-N tokens by H (N=10, minimum spacing so forks don't cluster in one identifier), plus 3 random positions as controls. Candidates are code tokens only; comment tokens are excluded (D-029, after the smoke run showed 64 % of high-entropy tokens sit in comments).
2. At each fork, force each of the top-3 alternatives that differ from the actual token, and complete the app from there (1 rollout per alternative, same sampling settings).
3. **Baseline:** at the same fork, keep the actual token and re-sample the continuation 2 times. This measures how much the outcome drifts from sampling noise alone.
4. Compare all rollouts' final apps (see 5.5).

**Fork importance** = mean outcome distance between alternative-token rollouts and same-token rollouts, minus the mean distance among same-token rollouts (noise floor). This separates "the model was undecided" (entropy) from "the choice mattered" (consequence). *(Builder-check call: without the baseline, sampling noise and choice are confounded.)*

**Outcome distance** (primary): screenshot-embedding distance, combined with a change in render status (works ↔ broken counts as maximal distance). Secondary, reported alongside: code-embedding distance and structural-fingerprint distance. Consequence: WP5 needs WP4's render and embedding tooling first.

Done on a subset first because of compute (see D2).

Optional (subset): Bigelow-style outcome tracking – at positions along the path, sample K continuations and track how the distribution over outcomes shifts; change points show where the decision effectively happened.

### 5.5 Outcome measures per app (automatic)

- **Renders**: loads in headless Chromium (Playwright) without uncaught errors; console error count.
- **Screenshot** after load (and after a short scripted interaction where the prompt allows, e.g. key presses for games).
- **Size**: chars, lines, tokens.
- **Structural fingerprint**: DOM tag histogram, JS function names, CSS colour palette.
- **Embeddings**: of source code and of screenshot (model chosen by the builder, documented), for distance computations.
- **Prompt adherence** (tight briefs only): small hand-written checklist per prompt (e.g. "has 10 questions", "shows score").

### 5.6 Novelty and usefulness

- **Novelty, automatic**: distance of an app to the other apps of the same prompt (ensemble rarity) and to the whole corpus, in embedding and structural space.
- **Novelty and usefulness, human**: blind rating page. Raters see one app (live, interactive) plus its prompt, rate *usefulness* (1–5), *novelty* ("I have seen something like this before" → "this is new to me", 1–5), and *works* (yes / partly / no). Order randomised, no model or entropy info shown.
- **Raters: the public, recruited via Reddit** (not students). Consequences for the page: publicly reachable without login; anonymous, no personal data collected (no names, emails, IP addresses stored); apps shown in a sandboxed iframe; one rating per app per browser; simple protection against spam/bot ratings; per-rating metadata limited to timestamp, app id, a random browser id and time spent on the app. Hosting: the project website (section 9) with a Cloudflare Worker + D1 backend. Short intro text explains the study; ratings are published as open data.

### 5.7 Analysis measures

Per app and per section:
- mean H, mean S, fork share (H ≥ 2 bits), fork density by section
- normalised per token, per character and per line (tokenizer effect)
- **path action** = Σ S (= −log₂ p of the whole app); comparison of an app's action with its siblings
- entropy profile over position (where in the file the forks sit)
- surprisal at forks (originality signal)

Across apps / prompts:
- distributions by openness level and domain
- correlation of landscape measures with renders / adherence / human usefulness / human novelty / ensemble rarity
- fork importance vs. entropy

Representation trajectory (subset): path of hidden states through the generation, speed and turning points, and whether turning points coincide with forks.

## 6. Hypotheses (phase A unless marked)

- **H1** Per-token entropy is strongly bimodal: few open positions, most near 0 bits.
- **H2** Openness of the brief shifts forks: tight briefs → forks in details (names, colours, constants), loose/free → high-entropy forks early and at the idea level, with larger ensemble spread.
- **H3** Entropy ≠ consequence: only part of the high-entropy forks change the outcome substantially; fork importance is better predicted by entropy plus position/section than by entropy alone.
- **H4** Landscape measures predict usefulness: apps that work and are rated useful differ in entropy profile (e.g. fewer late high-surprisal tokens in JS).
- **H5** Landscape measures predict novelty: surprisal at forks and path action correlate with ensemble rarity and human novelty ratings.
- **H6 (phase B)** Over crafting iterations, entropy at the forks decreases as decisions settle; late increases mark a change of course.

These are framed to be falsifiable; null results are results and get documented.

**Analysis plan frozen before WP3** *(builder-check call)*: before the main grid is generated, `METHODS.md` fixes for each hypothesis the primary outcome, the statistical model and what counts as support. Defaults: samples are nested in prompts, so cross-app effects use mixed models with prompt as random effect (domain and openness as fixed effects); within-prompt correlations are reported as well; Holm correction across the H1–H5 primary tests. Primary outcomes: H4 → human usefulness rating (renders/adherence secondary); H5 → human novelty rating (ensemble rarity secondary). This is a lightweight internal pre-registration (a dated, committed file), not a formal registry entry (decided 2026-11-19). Exploratory analyses are allowed but labelled as such.

## 7. Phase B: crafting loop

Same recorder, plus a loop:
1. Model writes draft k (logged as in 5.3).
2. Draft is rendered; feedback is built from console errors and the screenshot (the 9B model is multimodal), optionally plus a fixed critique prompt.
3. Model writes draft k+1 given prompt, draft k and feedback (logged).
4. Stop after a fixed number of rounds or when the model declares done.

Extra measures: bits added per iteration (surprisal of draft k+1 given draft k), diff size and location, whether changed spans sat at high-entropy positions of draft k, landscape change between rounds (KL between next-token distributions at aligned positions). Details of feedback design: D5. Phase B is designed in detail only after phase A shows whether there is a pattern.

## 8. Compute plan: machines and run window

| Machine | Resources (checked 2026-11-19) | Role |
|---|---|---|
| **Mac Studio** (M2 Max) | 32 GB unified memory, ~690 GB free on `/Volumes/Data` | **Main series**: all generation and fork experiments of the primary condition (Qwen3.5-9B, bf16) |
| **Desktop PC** (RTX 4060 Ti) | 16 GB VRAM, WSL Ubuntu with 15 GB RAM, ~900 GB free disk | **Work that does not mix with the main series**: rendering, screenshots, embeddings (WP4); optional separate condition "Qwen3.5-9B, 8-bit" to test whether quantization changes the landscape (WP9) |

Rules:
- **One condition = one machine + runtime + precision.** Logits differ slightly between MLX/Metal and CUDA and between precisions, so a condition is never split across machines. The desktop is **not** a fallback for the Studio's main series; if the Studio is unavailable, the series pauses.
- Qwen3.5-9B in bf16 (~18 GB) does not fit into the desktop's 16 GB VRAM, which is why the main series runs on the Studio.
- **Run window: 23:00–06:00** on both machines, started by launchd (Studio) and a scheduled task (desktop). Ollama on the Studio is stopped at the start of the window and restarted at the end.
- Jobs are **resumable**: the unit of work is one app (or one fork rollout). A unit that is interrupted at 06:00 is discarded and redone in the next window; finished units are never recomputed. A run may span many nights without affecting consistency.
- Throughput is measured in WP1 (single sequence and batched); the time table below is replaced by measured values in `JOURNAL.md`.

Estimate from measured throughput (assumes 17.3 tok/s single-sequence decode as measured in generation with the float32 logits head, 4,176 tokens per app as in the smoke app, forks at 32 rollout-equivalents per app as in the earlier estimate, no batching; real prompts with open scope may yield longer apps):

| Task | Compute | Nights (7 h) |
|---|---|---|
| Main grid, 420 apps | ~28 h | ~4 |
| Fork experiments, 2 apps per prompt (42 apps, with baseline) | ~90 h | ~13 |
| Fork experiments, all 420 apps | ~900 h | not planned; subset only (D2) |

## 9. Project website (GitHub Pages)

Public site of the study, served from the public repo via GitHub Pages. Static pages; ratings go to a separate backend.

```
entropytrail (GitHub Pages)
├─ Home         What the project is about: the landscape idea, method in 3 steps
│               Key numbers (apps built, ratings received), top findings
├─ Apps         Gallery, filter by domain (education/game/tool/free) and openness
│  └─ App view  Try it (sandboxed iframe) → rate: usefulness · new? · works?
│               after rating: source code coloured by entropy, forks marked
└─ Findings     Results per hypothesis, charts (filled after WP6)
```

- **Blindness:** an app's measurements (entropy colouring, forks, any metric) are shown only *after* the visitor has rated it – this also rewards rating. Findings that could bias raters (e.g. "high entropy = more original") stay off the site until the rating phase is closed; until then Home shows only the project description and counts.
- **Which apps are shown:** the rating sample (D4). The site is built from the derived data in the repo (`results/`, app files copied into `site/apps/`), never from raw data on the Studio.
- **Ratings backend: Cloudflare Worker + D1** (decided 2026-11-19). No login; Turnstile bot protection; constraints from 5.6 apply (anonymous, no personal data, no IP storage, one rating per app per browser). The Worker exposes: submit rating, and aggregate counts for the Home page. A scheduled export writes the ratings as open data (CSV) into the repo.
- **Lifecycle:** (1) project description only, (2) rating phase opens after WP3/WP4, announced on Reddit, (3) after the rating phase closes, Findings and full metrics are published.

## 10. Repository and documentation

```
entropytrail/
  README.md                  what this is, how to run, where data lives
  docs/
    superpowers/specs/       design docs (this file)
    DECISIONS.md             numbered decision log (date, decision, reason, alternatives)
    DATA.md                  data dictionary: every file, field, unit, definition
    JOURNAL.md               dated run journal: what was run, on what, what happened
    METHODS.md               measure definitions and formulas (kept in sync with code)
  prompts/prompts.yaml       versioned prompt set + system prompt
  src/entropytrail/          recorder, generation, forks, evaluation, analysis
  notebooks/                 analysis notebooks, each tied to a hypothesis
  site/                      GitHub Pages site (section 9)
  worker/                    Cloudflare Worker for ratings
  results/                   small derived tables and figures (committed)
```

Raw data: `/Volumes/Data/entropytrail/<condition>/<prompt_id>/<sample>/` with `app.html`, `tokens.parquet`, `meta.json` (model id + weights hash, code commit, seed, settings, timings), optional `hidden.npy`, `forks/…`, `render/…`.

Every run writes a manifest; the journal links manifests. A "condition" = model + precision + sampling settings + system prompt version; any change creates a new condition.

## 11. Work packages

| WP | Content | Done when |
|---|---|---|
| WP0 | Repo, docs skeleton (README, DECISIONS, DATA, JOURNAL, METHODS), Python env on Studio and desktop, night scheduling (launchd / scheduled task) with Ollama stop/start, data transfer Studio → desktop | skeleton committed, env reproducible |
| WP1 | Model download + runtime; recorder that logs 5.3 fields; validation: logged H/S match a token-by-token (chunk=1) teacher-forced recomputation within 1e-5 bits (decision D-021; chunked recomputation differs by up to ~0.25 bits of bf16 noise and is informational only), and the same seed reproduces the same token sequence on the same runtime (bit-exact logits not required) | one app generated and logged, validation passes |
| WP2 | Prompt set (~21 prompts, system prompt), reviewed by Robert; analysis plan frozen in `METHODS.md` | `prompts.yaml` v1 and analysis plan frozen |
| WP3 | Batch generation: 20 samples × all prompts, resumable | ~420 apps with logs |
| WP4 | Automatic evaluation on the desktop (render, errors, screenshots, fingerprints, embeddings, adherence checklists) | table of outcomes for all apps |
| WP5 | Fork experiments, subset first, then scale per D2 | fork-importance table |
| WP6 | Analysis + visualisation: entropy-coloured source viewer, profiles, comparisons by cell; tests of H1–H5 | notebooks + figures per hypothesis |
| WP7 | Website (section 9): site skeleton, app gallery + rating flow, Cloudflare Worker + D1, Reddit announcement, data collection, rating export | site live, ratings for the sampled apps |
| WP8 | Phase B: crafting loop on the same recorder, design after WP6 results | iterative runs logged, H6 analysed |
| WP9 | Optional desktop condition: Qwen3.5-9B 8-bit on CUDA, subset of the grid (quantization effect on the landscape) | second condition analysed against the bf16 series |
| WP10 | Comparison model (Qwen3.6-35B-A3B) on a subset | second condition analysed |
| WP11 | Write-up: report; Findings section of the website filled | report and findings published |

Order: WP0 → WP1 → WP2 → WP3 → (WP4, WP5, WP7 in parallel) → WP6 → WP8. WP9 and WP10 when the respective machine has free windows.

## 12. Open decisions

| ID | Question | Current proposal |
|---|---|---|
| D1 | Reasoning/thinking mode on or off? Thinking tokens are part of the landscape but make runs much longer. | Off for the main grid; one extra condition with thinking on for the free prompts |
| D2 | Fork-experiment budget: N forks × alternatives × rollouts × apps | Start with 3 prompts × 5 apps, measure time, then decide scale |
| D3 | Hidden states: which layers, which subset of apps | 3 layers (early/middle/late) for 1 sample per prompt |
| D4 | ~~Who rates~~ → public via Reddit (decided). Still open: ratings per app, all apps or a sample | ≥3 ratings per app on a stratified sample of ~100 apps; extend if response is good |
| D5 | Phase B feedback: automatic only (errors + screenshot), fixed critique prompt, or human feedback? | Decide after WP6 |
| D6 | Runtime: MLX vs PyTorch-MPS for the 9B model | Builder's call, but must give full-precision logits and hidden-state access; document in DECISIONS |
| D7 | ~~Repo visibility~~ → public from the start (decided 2026-11-19): needed for GitHub Pages on a free account, and makes the frozen analysis plan publicly timestamped. Raw data stays off GitHub. | – |
| D8 | Language of prompts (English vs German) | English for the main grid; German variants as an optional later condition |
| D9 | Temperature sweep after the first DoE (agreed with Robert 2026-11-20) | After the T = 1.0 grid is complete and analysed: the same 21 prompts × 20 samples at lower temperatures (values to fix then, e.g. 0.7 and 0.5), each a separate condition_id. The recorder logs the untempered distribution, so H and S stay comparable across temperatures; only the sampled paths change. Parked alternative: generate per prompt until 20 syntactically valid apps exist (cap 60, all attempts kept). |

## 13. Left to the builder

- Exact library choices (Playwright, parquet, embedding models), file formats beyond what 5.3 and 9 specify.
- Fork-spacing rule details, section classifier implementation.
- Visual design of the website, viewer and rating flow; static-site tooling (plain HTML or a simple generator).
- Each such choice is recorded in `DECISIONS.md`.
