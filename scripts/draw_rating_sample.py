"""Draw the human rating sample (D-043): 5 apps per prompt of the main grid (T = 1.0), seeded.

    uv run --group analysis python scripts/draw_rating_sample.py

Writes results/rating-sample.csv (prompt_id, sample). Refuses to overwrite: the sample is
fixed before any human rating exists. Apps are drawn uniformly within each prompt, whether
they render or not (a broken app is a valid object to rate).
"""
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED, PER_PROMPT = 43, 5


def draw(outcomes: pd.DataFrame, seed: int = SEED, k: int = PER_PROMPT) -> pd.DataFrame:
    rows = []
    for pid in sorted(outcomes["prompt_id"].unique()):
        samples = sorted(outcomes.loc[outcomes["prompt_id"] == pid, "sample"].astype(int))
        for s in sorted(random.Random(f"{seed}:{pid}").sample(samples, k)):
            rows.append({"prompt_id": pid, "sample": s})
    return pd.DataFrame(rows)


def main() -> int:
    out = ROOT / "results" / "rating-sample.csv"
    if out.exists():
        print(f"{out} exists; the sample is fixed", file=sys.stderr)
        return 1
    df = draw(pd.read_csv(ROOT / "results" / "outcomes.csv", dtype={"sample": int}))
    df.to_csv(out, index=False)
    print(len(df), "apps,", df["prompt_id"].nunique(), "prompts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
