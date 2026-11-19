# Foundation & Recorder Implementation Plan (Plan 1 of the study)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A documented, tested Python package that generates an HTML app with Qwen3.5-9B (bf16, MLX) on the Mac Studio while logging the full next-token statistics per token, validates those statistics, runs resumably in the 23:00–06:00 window and stops/restarts Ollama around it.

**Architecture:** `entropytrail` package (uv, Python 3.12). A pure-function stats layer turns full log-prob vectors into per-token records; a recorder wraps `mlx_lm.generate.generate_step` (which yields the full log-prob vector per step) and samples at T=1.0 with no truncation; a teacher-forcing scorer recomputes the same stats in chunks for validation; a runner processes (prompt, sample) units into atomic run directories and aborts at a deadline. Tests use a tiny randomly initialised `qwen3_5` model built from the real config, so the 9B weights are only needed for the final validation task.

**Tech Stack:** Python 3.12 via uv, mlx + mlx-lm 0.32.2, numpy, pyarrow (parquet), pyyaml, pytest. launchd for scheduling.

**Spec:** `docs/superpowers/specs/2026-11-19-entropytrail-design.md` (sections 4, 5.2, 5.3, 8, 10; work packages WP0 and WP1).

**Scope of this plan:** WP0 (Studio side) + WP1. Not in this plan, each gets its own plan: prompt grid v1 and frozen analysis plan (WP2), batch generation of the grid (WP3), desktop environment + evaluation + Studio→desktop transfer (WP4), forks (WP5), analysis incl. the `section` field and hidden states (WP6, D3), website + Cloudflare Worker (WP7), phase B (WP8), 8-bit condition (WP9), comparison model (WP10).

## Global Constraints

- Primary condition: `Qwen/Qwen3.5-9B`, **bf16**, MLX, mlx-lm **0.32.2** pinned exactly; HF cache at `HF_HOME=/Volumes/Models/huggingface` (weights present, 4 safetensors, 19.3 GB).
- Sampling: **temperature 1.0, no top-p/top-k truncation**; seed recorded per sample; max output **16,000** tokens; truncated runs kept and flagged (`finish_reason: "max_tokens"`).
- Thinking mode **off** for the primary condition (D1); passed as `enable_thinking=False` to the chat template.
- All information quantities in **bits** (log base 2). Top-k stored with **k = 20**.
- Full logits are never written to disk.
- Raw data root: `/Volumes/Data/entropytrail/` (configurable via env `ENTROPYTRAIL_DATA`); layout `<condition_id>/<prompt_id>/<sample:03d>/` with `app.html`, `tokens.parquet`, `meta.json`.
- **One condition = one machine + runtime + precision**; any change of model, precision, sampling, system prompt version or thinking creates a new `condition_id`.
- Run window **23:00–06:00**; a unit interrupted by the deadline is discarded, finished units are never recomputed.
- Validation (decision D-021): logged H and S equal a **token-by-token (chunk=1) teacher-forced recomputation, max absolute difference ≤ 1e-5 bits**; chunked recomputation is reported as informational noise only; same seed ⇒ same token sequence on the same runtime (bit-exact logits not required).
- Repo is public: no `Claude-Session:` trailers, no claude.ai links; commit trailer only `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Every choice the builder makes is recorded in `docs/DECISIONS.md`.

## Review Focus

1. **Model emits no HTML or wraps it in markdown fences / prose** – expected: `extract_html` returns the HTML inside fences, `None` when there is no `<html`/`<!doctype`; the run is still stored with `html_extracted: false`. → tests in Task 4.
2. **Deadline hits mid-generation** – expected: the partial directory is deleted, nothing half-written remains, the unit is retried next window. → test in Task 8.
3. **Process killed hard (power loss) leaving a `.partial` directory** – expected: next run removes stale `.partial` dirs and redoes the unit. → test in Task 8.
4. **Very long outputs (up to 16k tokens) in teacher forcing** – expected: memory bounded by chunking, results independent of chunk size. → test in Task 7.
5. **Ollama restart fails to happen if the night job crashes** – expected: Ollama is restarted on every exit path of the night script. → test in Task 10.

---

### Task 1: Repository skeleton, environment and documentation skeleton

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `README.md`
- Create: `docs/DECISIONS.md`, `docs/DATA.md`, `docs/JOURNAL.md`, `docs/METHODS.md`
- Create: `src/entropytrail/__init__.py`, `tests/test_smoke.py`

**Interfaces:**
- Produces: installable package `entropytrail` with `__version__ = "0.1.0"`; console script `et = "entropytrail.cli:main"` (cli module arrives in Task 8); pytest marker `slow` (tests needing the 9B weights).

- [ ] **Step 1:** `.python-version` = `3.12`. `pyproject.toml`: hatchling build, `requires-python = ">=3.12,<3.13"`, dependencies `mlx-lm==0.32.2`, `numpy`, `pyarrow`, `pyyaml`; dev group `pytest`; `[tool.pytest.ini_options]` with `markers = ["slow: needs Qwen3.5-9B weights"]` and `addopts = "-m 'not slow'"`.
- [ ] **Step 2:** `.gitignore`: `.venv/`, `__pycache__/`, `*.partial/`, `.pytest_cache/`, `data/`.
- [ ] **Step 3:** Write `tests/test_smoke.py::test_version` asserting `entropytrail.__version__ == "0.1.0"`; run `uv sync && uv run pytest` → FAIL (no package), then add `src/entropytrail/__init__.py`, rerun → PASS.
- [ ] **Step 4:** Docs skeleton:
  - `README.md`: one paragraph on the study (from spec §1), link to the spec, setup (`uv sync`), where raw data lives, how to run tests.
  - `docs/DECISIONS.md`: format `## D-NNN · YYYY-MM-DD · title` with *Decision / Reason / Alternatives*. Seed it with the decisions already made in the spec: approach C; Qwen3.5-9B bf16 primary; local only, no Ollama for measurement; prompt grid 3×2+free; 20 samples, T=1.0, no truncation; thinking off (D1 proposal); top-20 stored, no full logits; fork baseline; analysis plan frozen before WP3 (light pre-registration); Studio main series / desktop separate; 23:00–06:00 window; public repo; Cloudflare Worker + D1 ratings; raters via Reddit; trez63 only inspiration; runtime MLX + mlx-lm 0.32.2 (D6, this plan).
  - `docs/DATA.md`: headings for each file in the run directory; filled in Task 5.
  - `docs/METHODS.md`: definitions of H, S, varentropy, margin (spec §5.3 formulas, bits); filled in Task 2.
  - `docs/JOURNAL.md`: format `## YYYY-MM-DD` entries; first entry "project set up".
- [ ] **Step 5:** Commit `chore: project skeleton and documentation skeleton`.

### Task 2: Token statistics

**Files:**
- Create: `src/entropytrail/stats.py`
- Test: `tests/test_stats.py`
- Modify: `docs/METHODS.md`

**Interfaces:**
- Produces:
  - `TOPK = 20`
  - `@dataclass(frozen=True) TokenStats: H: float; S: float; varentropy: float; margin: float; topk_ids: np.ndarray[int32, (20,)]; topk_logp2: np.ndarray[float32, (20,)]`
  - `token_stats(logprobs: mx.array, token_id: int) -> TokenStats` — `logprobs` is a 1-D natural-log probability vector over the vocabulary (as yielded by `generate_step`); computed in float32 on device, top-k sorted descending.
  - `token_stats_batch(logprobs: mx.array, token_ids: mx.array) -> dict[str, np.ndarray]` — same for a `(T, V)` block, keys `H, S, varentropy, margin, topk_ids, topk_logp2`; used by teacher forcing.

- [ ] **Step 1: Failing tests**

```python
def lp(p): return mx.log(mx.array(p, dtype=mx.float32))

def test_uniform_four():
    s = token_stats(lp([.25,.25,.25,.25]), 2)
    assert s.H == pytest.approx(2.0, abs=1e-5)
    assert s.S == pytest.approx(2.0, abs=1e-5)
    assert s.varentropy == pytest.approx(0.0, abs=1e-5)
    assert s.margin == pytest.approx(0.0, abs=1e-5)

def test_skewed():
    s = token_stats(lp([.5,.25,.125,.125]), 3)
    assert s.H == pytest.approx(1.75, abs=1e-5)
    assert s.S == pytest.approx(3.0, abs=1e-5)
    assert s.varentropy == pytest.approx(0.6875, abs=1e-5)   # E[(-log2p)^2] - H^2 = 3.75 - 3.0625
    assert s.margin == pytest.approx(1.0, abs=1e-5)

def test_topk_sorted_and_padded_shape():
    v = mx.log(mx.softmax(mx.random.normal((1000,), key=mx.random.key(0))))
    s = token_stats(v, 5)
    assert s.topk_ids.shape == (20,) and np.all(np.diff(s.topk_logp2) <= 0)
    assert s.topk_ids[0] == int(mx.argmax(v))

def test_zero_probability_entries_do_not_nan():
    s = token_stats(mx.array([0.0, -mx.inf, -mx.inf, -mx.inf]), 0)
    assert s.H == 0.0 and s.S == 0.0

def test_batch_matches_single():
    v = mx.log(mx.softmax(mx.random.normal((7, 500), key=mx.random.key(1)), axis=-1))
    ids = mx.array([3,1,4,1,5,9,2])
    b = token_stats_batch(v, ids)
    for t in range(7):
        s = token_stats(v[t], int(ids[t]))
        assert b["H"][t] == pytest.approx(s.H, abs=1e-5) and b["S"][t] == pytest.approx(s.S, abs=1e-5)
```

Vocabulary smaller than 20 in the first tests: top-k is `min(20, V)` there; the shape test uses V=1000.

- [ ] **Step 2:** `uv run pytest tests/test_stats.py -v` → FAIL (import error).
- [ ] **Step 3:** Implement both functions. Treat `-inf` log-probs as contributing 0 to H (`where(p>0, …, 0)`); convert nat → bits by `/ ln 2`; top-k via `mx.argpartition` then sort.
- [ ] **Step 4:** Tests PASS.
- [ ] **Step 5:** Write the four definitions with formulas into `docs/METHODS.md`. Commit `feat: per-token entropy, surprisal, varentropy, margin, top-k`.

### Task 3: Conditions, prompts and seeds

**Files:**
- Create: `src/entropytrail/condition.py`, `configs/conditions/qwen35-9b-bf16-mlx.yaml`, `prompts/prompts.yaml`
- Test: `tests/test_condition.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Condition: name: str; model_repo: str; precision: str; runtime: str; runtime_version: str; temperature: float; max_tokens: int; thinking: bool; system_prompt_version: str; machine: str`
  - `Condition.condition_id -> str` — `f"{name}-{sha256(canonical json of all fields)[:8]}"`
  - `load_condition(path: Path) -> Condition`
  - `@dataclass(frozen=True) Prompt: id: str; domain: str; openness: str; text: str`
  - `@dataclass(frozen=True) PromptSet: version: str; system_prompts: dict[str, str]; prompts: list[Prompt]`
  - `load_prompts(path: Path) -> PromptSet`
  - `derive_seed(condition_id: str, prompt_id: str, sample: int) -> int` — first 4 bytes of sha256 of `"{condition_id}|{prompt_id}|{sample}"` as unsigned int.

- [ ] **Step 1:** `configs/conditions/qwen35-9b-bf16-mlx.yaml`: `name: qwen35-9b-bf16-mlx, model_repo: Qwen/Qwen3.5-9B, precision: bf16, runtime: mlx-lm, runtime_version: 0.32.2, temperature: 1.0, max_tokens: 16000, thinking: false, system_prompt_version: sys-v1, machine: mac-studio-m2max`.
- [ ] **Step 2:** `prompts/prompts.yaml` v0 (the full grid is WP2): `version: v0-smoke`; `system_prompts: {sys-v1: "<text>"}` where the text states: produce exactly one self-contained HTML file, inline CSS and JavaScript, no external resources or libraries, output only the file; one prompt `smoke-tool-01` (domain `tool`, openness `tight`): "Build a unit converter for centimetres, inches and feet."
- [ ] **Step 3: Failing tests**

```python
def test_condition_id_stable_and_sensitive(tmp_path):
    c = load_condition(Path("configs/conditions/qwen35-9b-bf16-mlx.yaml"))
    assert c.condition_id == load_condition(Path("configs/conditions/qwen35-9b-bf16-mlx.yaml")).condition_id
    assert c.condition_id.startswith("qwen35-9b-bf16-mlx-")
    assert dataclasses.replace(c, temperature=0.7).condition_id != c.condition_id

def test_prompts_load():
    ps = load_prompts(Path("prompts/prompts.yaml"))
    assert ps.prompts[0].id == "smoke-tool-01" and "sys-v1" in ps.system_prompts

def test_prompt_ids_unique(tmp_path):  # duplicate ids → ValueError
    ...

def test_seed_deterministic_and_distinct():
    assert derive_seed("c", "p", 0) == derive_seed("c", "p", 0)
    assert len({derive_seed("c", "p", i) for i in range(100)}) == 100
```

- [ ] **Step 4:** Implement; tests PASS. Commit `feat: conditions, prompt set, seed derivation`.

### Task 4: HTML extraction

**Files:**
- Create: `src/entropytrail/extract.py`
- Test: `tests/test_extract.py`

**Interfaces:**
- Produces: `extract_html(text: str) -> str | None`

- [ ] **Step 1: Failing tests** — cases: plain document starting with `<!DOCTYPE html>` → returned unchanged (stripped); document inside ```` ```html … ``` ```` with prose before/after → inner HTML only; fence without language tag → inner HTML; two fenced blocks → the first one containing `<html` or `<!doctype` (case-insensitive); text without any `<html`/`<!doctype` → `None`; unterminated fence (truncated run) → content after the opening fence.
- [ ] **Step 2:** FAIL → implement with a regex over fenced blocks, fallback to slicing from the first `<!doctype`/`<html` to the last `</html>` (or end) → PASS.
- [ ] **Step 3:** Commit `feat: extract HTML from model output`.

### Task 5: Run storage

**Files:**
- Create: `src/entropytrail/storage.py`
- Test: `tests/test_storage.py`
- Modify: `docs/DATA.md`

**Interfaces:**
- Consumes: `TokenStats` (Task 2), `extract_html` (Task 4).
- Produces:
  - `@dataclass TokenRecord: pos: int; token_id: int; text: str; char_offset: int; stats: TokenStats`
  - `@dataclass Recording: prompt_ids: list[int]; tokens: list[TokenRecord]; text: str; finish_reason: Literal["eos","max_tokens"]; seconds: float`
  - `data_root() -> Path` — env `ENTROPYTRAIL_DATA` or `/Volumes/Data/entropytrail`
  - `run_dir(root: Path, condition_id: str, prompt_id: str, sample: int) -> Path` — `root/condition_id/prompt_id/f"{sample:03d}"`
  - `write_run(final_dir: Path, rec: Recording, meta: dict) -> None` — writes into `final_dir.with_suffix(".partial")`, then `os.rename` to `final_dir`; files: `tokens.parquet` (columns `pos int32, token_id int32, text string, char_offset int32, H float32, S float32, varentropy float32, margin float32, topk_ids list<int32>, topk_logp2 list<float32>`), `output.txt` (raw model text), `app.html` (only if extraction succeeded), `meta.json` (given meta + `n_tokens`, `finish_reason`, `seconds`, `tokens_per_s`, `html_extracted`, `status: "done"`).
  - `is_done(final_dir: Path) -> bool` — `meta.json` exists with `status == "done"`.
  - `read_tokens(final_dir: Path) -> pyarrow.Table`
  - `clean_partials(root: Path) -> int` — deletes every `*.partial` dir below root, returns count.

- [ ] **Step 1: Failing tests** — round trip of a synthetic 3-token `Recording` (values and dtypes as listed, `topk_ids` length 20); `is_done` false before / true after; no `.partial` left after `write_run`; output without HTML → no `app.html`, `html_extracted: false`; `clean_partials` removes a planted `x.partial` dir and returns 1; `char_offset` of token t equals `len("".join(texts[:t]))`.
- [ ] **Step 2:** FAIL → implement → PASS.
- [ ] **Step 3:** Fill `docs/DATA.md`: every file, every column/key, unit, definition. Commit `feat: atomic run storage`.

### Task 6: Model loading, provenance and the recorder

**Files:**
- Create: `src/entropytrail/model.py`, `src/entropytrail/recorder.py`
- Create: `tests/conftest.py` (tiny-model fixture), `tests/test_recorder.py`

**Interfaces:**
- Consumes: `Condition`, `token_stats`, `TokenRecord`, `Recording`.
- Produces:
  - `load_model(cond: Condition) -> tuple[nn.Module, TokenizerWrapper]` — `mlx_lm.load(cond.model_repo)`; asserts weights dtype bf16 for `precision == "bf16"`.
  - `weights_fingerprint(repo: str) -> str` — sha256 over the sorted HF blob names of the snapshot's `*.safetensors` symlink targets (blob names are the files' sha256), no re-hashing of 19 GB.
  - `provenance() -> dict` — git commit + dirty flag, versions of mlx, mlx-lm, python, macOS, hostname.
  - `build_prompt_ids(tokenizer, system: str, user: str, thinking: bool) -> list[int]` — `apply_chat_template(..., add_generation_prompt=True, enable_thinking=thinking)`.
  - `class Interrupted(Exception)`
  - `tokenizer_io(tokenizer) -> tuple[set[int], Callable[[list[int]], list[str]]]` — EOS ids (`tokenizer.eos_token_ids`) and a `token_texts` function returning one string per token whose concatenation equals `tokenizer.decode(ids)` (built from the streaming detokenizer's segments, so multi-byte UTF-8 pieces are handled).
  - `generate_recorded(model, prompt_ids: list[int], *, seed: int, temperature: float, max_tokens: int, eos_ids: set[int], token_texts: Callable[[list[int]], list[str]], should_stop: Callable[[], bool] | None = None) -> Recording` — `mx.random.seed(seed)`; iterate `mlx_lm.generate.generate_step(mx.array(prompt_ids), model, max_tokens=max_tokens, sampler=make_sampler(temp=temperature))`, call `token_stats(logprobs, token)` per step; stop at the first token in `eos_ids` (the EOS token is recorded in `tokens` but contributes no text); `text = "".join(token_texts(ids))`; check `should_stop()` every step and raise `Interrupted`.
- Test fixture (`conftest.py`): `tiny_model` builds `mlx_lm.models.qwen3_5.Model(ModelArgs.from_dict(cfg))` from the real 9B `config.json` with overrides `hidden_size=64, intermediate_size=128, num_hidden_layers=4, layer_types=first 4, num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16, vocab_size=1000`, random weights after `mx.random.seed(0)` (verified to run with mlx-lm 0.32.2). Fixture `fake_io`: `eos_ids={999}`, `token_texts=lambda ids: [f"<{i}>" for i in ids]`.

- [ ] **Step 1: Failing tests** (tiny model + `fake_io`, prompt `[1,2,3]`, `max_tokens=50`)
  - `test_lengths_and_ranges`: `len(rec.tokens) <= 50`, every `0 <= H <= log2(1000)+1e-4`, `S >= 0`.
  - `test_surprisal_consistent_with_topk`: where the sampled token is in `topk_ids`, `S == -topk_logp2[idx]` (1e-4).
  - `test_same_seed_same_tokens`: two runs seed 7 → identical `token_id` sequences; seed 8 → different.
  - `test_should_stop_raises`: `should_stop` true after 5 calls → `Interrupted`.
  - `test_max_tokens_flag`: no EOS reachable → `finish_reason == "max_tokens"`.
  - `@pytest.mark.slow test_real_tokenizer_chat_template`: `build_prompt_ids` with the 9B tokenizer and `thinking=False` ends with the assistant generation prefix and contains no `<think>` opening the answer.
  - `test_weights_fingerprint_stable` (`slow`): same value twice, 64 hex chars.
- [ ] **Step 2:** FAIL → implement → PASS (`uv run pytest -v` and `uv run pytest -m slow -v`).
- [ ] **Step 3:** Commit `feat: recorder over generate_step with per-token statistics`.

### Task 7: Teacher-forcing scorer

**Files:**
- Create: `src/entropytrail/teacher_forcing.py`
- Test: `tests/test_teacher_forcing.py`

**Interfaces:**
- Consumes: `token_stats_batch`, tiny-model fixture, `generate_recorded`.
- Produces: `score_tokens(model, prompt_ids: list[int], output_ids: list[int], chunk: int = 512) -> dict[str, np.ndarray]` — one prefill of the prompt with `mlx_lm.models.cache.make_prompt_cache(model)`, then feeds `output_ids` in chunks of `chunk` through `model(x, cache=cache)`; the distribution for output position t comes from the logits at the previous position (last prompt position for t=0); returns `token_stats_batch` arrays of length `len(output_ids)`. Memory: only one `(chunk, V)` block of log-probs is alive at a time.
- Also produces: `compare(rec: Recording, scored: dict) -> dict` — keys `p99_abs_dH`, `p99_abs_dS`, `max_abs_dH`, `max_abs_dS`, `ok` (`p99 ≤ 0.01` for both).

- [ ] **Step 1: Failing tests**
  - `test_matches_recorder`: tiny model, 200 generated tokens; `compare(...)["ok"]` is True.
  - `test_chunk_size_invariance`: `chunk=7` vs `chunk=512` → H arrays equal within 1e-4.
  - `test_long_sequence_bounded`: 3,000 random output ids with `chunk=256` completes and returns length 3,000 (guards the 16k case without the 9B model).
- [ ] **Step 2:** FAIL → implement → PASS.
- [ ] **Step 3:** Commit `feat: chunked teacher-forcing scorer and validation comparison`.

### Task 8: Runner and CLI

**Files:**
- Create: `src/entropytrail/runner.py`, `src/entropytrail/cli.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `@dataclass Unit: prompt: Prompt; sample: int`
  - `plan_units(ps: PromptSet, n_samples: int, prompt_ids: list[str] | None) -> list[Unit]` — prompt-major order.
  - `run_units(units, cond, ps, model, tokenizer, root: Path, deadline: datetime | None, *, fingerprint: str, now: Callable[[], datetime] = datetime.now, io: tuple | None = None, prompt_ids_fn: Callable[[Prompt], list[int]] | None = None) -> dict` — `io` defaults to `tokenizer_io(tokenizer)`, `prompt_ids_fn` to `build_prompt_ids` with the condition's system prompt and thinking flag; tests inject `fake_io` and a fixed `prompt_ids_fn`. — `clean_partials(root/cond.condition_id)` first; skips `is_done` units; before each unit and inside generation (`should_stop`) checks `now() >= deadline`; on `Interrupted` removes the unit's `.partial` dir and stops; meta per unit = provenance + `condition` (asdict) + `condition_id` + `prompt` (asdict) + `prompt_set_version` + `system_prompt_version` + `sample` + `seed` + `weights_fingerprint` (the passed `fingerprint`) + `started_at`/`finished_at` (ISO). Appends one line per call to `root/runlog.jsonl`: `{started, ended, condition_id, done, skipped, aborted}`. Returns that dict.
  - CLI `et` (argparse):
    - `et generate --condition PATH --prompts PATH --prompt-id ID --sample N` → one unit, no deadline.
    - `et run-window --condition PATH --prompts PATH --samples N [--prompt-ids …] --until HH:MM` → deadline = next occurrence of HH:MM.
    - `et validate --condition PATH --prompts PATH --tokens 1500` → generates one unit into `root/_validation/…`, runs `score_tokens`, then regenerates with the same seed; prints and writes `validation.json` (`compare` result + `same_tokens: bool` + throughput). Exit code 1 if not ok.
    - `et bench --condition PATH --tokens 1000` → single-sequence tokens/s (prefill and decode separately), printed and written to `root/_bench/<timestamp>.json`.
- [ ] **Step 1: Failing tests** (tiny model, `tmp_path` as root, fake clock)
  - `test_resume_skips_done`: 3 units, run, run again → second call `done == 0, skipped == 3`.
  - `test_deadline_mid_generation_discards_partial`: clock passes the deadline after N steps → `aborted == 1`, no `.partial` dir, unit not done; next call without deadline completes it.
  - `test_stale_partial_cleaned`: planted `…/000.partial` is removed and the unit is produced.
  - `test_runlog_appended`: one JSON line per call with the listed keys.
  - `test_meta_has_provenance`: meta contains `seed == derive_seed(...)`, `condition_id`, `weights_fingerprint`, git commit.
- [ ] **Step 2:** FAIL → implement → PASS. The tiny model enters `run_units` via injected `model`, `io`, `prompt_ids_fn` and `fingerprint="test"`; the CLI is a thin wrapper and is exercised in Task 9.
- [ ] **Step 3:** Commit `feat: resumable runner with deadline and et CLI`.

### Task 9: Validation and benchmark on Qwen3.5-9B (real model)

**Files:**
- Modify: `docs/JOURNAL.md`, `docs/DECISIONS.md`, `docs/superpowers/specs/2026-11-19-entropytrail-design.md` (§8 time table)

Ollama must be stopped first (it may hold a model in memory): this needs Robert's `sudo` — ask him to run `! sudo launchctl bootout system/com.robert.ollama`, and afterwards `! sudo launchctl bootstrap system /Library/LaunchDaemons/com.robert.ollama.plist`.

- [ ] **Step 1:** `uv run pytest -m slow -v` → PASS.
- [ ] **Step 2:** `uv run et bench --condition configs/conditions/qwen35-9b-bf16-mlx.yaml --tokens 1000` → record prefill and decode tokens/s. Expected order of magnitude: 15–25 tok/s decode.
- [ ] **Step 3:** `uv run et validate --condition … --prompts prompts/prompts.yaml --tokens 1500` → exit 0, `ok: true`, `same_tokens: true` (criterion per D-021; the chunked p99 is informational).
- [ ] **Step 4:** `uv run et generate … --prompt-id smoke-tool-01 --sample 0` → full app; open `app.html` in a browser and check it renders; check `meta.json` and `tokens.parquet` against `docs/DATA.md`.
- [ ] **Step 5:** JOURNAL entry with bench numbers, validation numbers, smoke app result. Replace the rough §8 time table in the spec with estimates from the measured throughput and the smoke app's token count. DECISIONS entry for the tolerance outcome and for "batched generation evaluated in the WP3 plan".
- [ ] **Step 6:** Commit `docs: validation and benchmark of the 9B recorder`.

### Task 10: Night scheduling on the Studio

**Files:**
- Create: `scripts/night-studio.sh`, `ops/launchd/de.entropytrail.night.plist`, `ops/README.md`
- Test: `tests/test_night_script.py`

**Interfaces:**
- Consumes: `et run-window`.
- Produces: root LaunchDaemon `de.entropytrail.night` starting `scripts/night-studio.sh` daily at 23:00 (`StartCalendarInterval` Hour 23 Minute 0), logs to `/Users/robert/Library/Logs/entropytrail-night.log`.

`night-studio.sh` (runs as root): `set -euo pipefail`; `trap restart_ollama EXIT` where `restart_ollama` runs `launchctl bootstrap system /Library/LaunchDaemons/com.robert.ollama.plist` (ignore "already loaded"); `launchctl bootout system/com.robert.ollama` (ignore "not loaded"); then `sudo -u robert -H env HF_HOME=/Volumes/Models/huggingface /opt/homebrew/bin/uv --directory /Users/robert/projects/entropytrail run et run-window --condition … --prompts … --samples "${ET_SAMPLES:-20}" --until 06:00`. The job's arguments come from `ops/night.env` (condition path, prompts path, samples), so the queue is changed without touching launchd. `ET_DRY_RUN=1` makes every external command an `echo`.

- [ ] **Step 1: Failing tests** — run the script with `ET_DRY_RUN=1` via `subprocess`: output contains `bootout system/com.robert.ollama`, the `et run-window … --until 06:00` line, and `bootstrap system …com.robert.ollama.plist` **last**; with a forced failure (`ET_DRY_RUN=1 ET_FAIL_RUN=1`) the script exits non-zero and the bootstrap line is still printed last.
- [ ] **Step 2:** FAIL → implement → PASS; `plutil -lint ops/launchd/de.entropytrail.night.plist` → OK.
- [ ] **Step 3:** `ops/README.md`: install/uninstall commands for Robert (`sudo cp … /Library/LaunchDaemons/ && sudo launchctl bootstrap system …`), how to change the queue in `ops/night.env`, how to check the log and `runlog.jsonl`, how to pause the series.
- [ ] **Step 4:** Ask Robert to install the daemon (sudo). After the first night: check log, `runlog.jsonl`, that Ollama is running again (`curl -s 127.0.0.1:11434/api/version`). JOURNAL entry.
- [ ] **Step 5:** Commit `feat: nightly run window with Ollama stop/restart`.
