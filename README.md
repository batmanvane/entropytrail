# Entropy Trail

Entropy Trail is a basic-research study of how decisions emerge inside a language model while it builds something. A local model (Qwen3.5-9B, bf16, MLX) writes small single-file HTML apps, and for every generated token we record the entropy of the next-token distribution, the surprisal of the sampled token, varentropy, the top-1/top-2 margin and the top-20 candidates. We ask whether the way the probability landscape reshapes shows a pattern during construction, and whether entropy (or a related quantity) indicates that an output is useful or novel.

Full design: [docs/superpowers/specs/2026-11-19-entropytrail-design.md](docs/superpowers/specs/2026-11-19-entropytrail-design.md)

## Setup

```
uv sync
```

Python 3.12 is provided by uv; `mlx-lm` is pinned to 0.32.2. The model weights are read from the Hugging Face cache (`HF_HOME`).

## Raw data

Raw data lives outside the repo in `/Volumes/Data/entropytrail/` (override with the environment variable `ENTROPYTRAIL_DATA`), laid out as `<condition_id>/<prompt_id>/<sample:03d>/` with `output.txt` (raw model output), `app.html` (extracted app, if any), `tokens.parquet` and `meta.json`. Each condition directory also holds `condition.json` (condition manifest, checked on every run); the data root holds `runlog.jsonl`, `_validation/` and `_bench/`. See [docs/DATA.md](docs/DATA.md).

## How to run

```
C=configs/conditions/qwen35-9b-bf16-mlx.yaml; P=prompts/prompts.yaml
uv run et generate   --condition $C --prompts $P --prompt-id smoke-tool-01 --sample 0
uv run et run-window --condition $C --prompts $P --samples 20 --until 06:00 [--prompt-ids ID ...]
uv run et validate   --condition $C --prompts $P --tokens 1500    # recorder validation (D-021)
uv run et bench      --condition $C --tokens 1000                 # single-sequence throughput
```

`generate` and `run-window` skip units that are already done and refuse to run on a git working tree with uncommitted changes unless `--allow-dirty` is given. Stop Ollama (or make sure it holds no model) before running the 9B model. The nightly schedule on the Studio is described in [ops/README.md](ops/README.md).

## Tests

```
uv run pytest              # fast tests
uv run pytest -m slow      # tests that need the Qwen3.5-9B weights
```

## Documentation

- [docs/DECISIONS.md](docs/DECISIONS.md): decision log
- [docs/METHODS.md](docs/METHODS.md): definitions of the recorded quantities
- [docs/DATA.md](docs/DATA.md): data formats
- [docs/JOURNAL.md](docs/JOURNAL.md): dated project journal
