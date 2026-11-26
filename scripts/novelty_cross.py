"""Cross-temperature blind novelty pass (D-041, exploratory).

    uv run python scripts/novelty_cross.py build      # writes the key (blind id -> condition, sample)
    uv run --group analysis python scripts/novelty_cross.py collect

`build` pools all apps of every temperature per prompt, shuffled with a seeded order, and writes
/Volumes/Data/entropytrail/_novelty-cross/key.json (agents never read it). Serve the pool with
`scripts/serve_apps.py --port 8771 --blind-key <key.json>`; agents (docs/novelty-cross-brief.md)
write /Volumes/Data/entropytrail/_novelty-cross/scores/<prompt_id>.json. `collect` maps the
scores back and writes results/temperature-sweep/novelty-cross.csv and novelty-cross.json.
"""

import argparse
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from entropytrail.novelty_cross import analyse, build_key, collect

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/Volumes/Data/entropytrail")
WORK = DATA / "_novelty-cross"
CONDITIONS = {1.0: "qwen35-9b-bf16-mlx-1195d88b", 0.7: "qwen35-9b-bf16-mlx-t0p7-a31fc218",
              0.5: "qwen35-9b-bf16-mlx-t0p5-c9bce32f"}


def prompts() -> list[dict]:
    return yaml.safe_load((ROOT / "prompts" / "prompts.yaml").read_text())["prompts"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "collect", "order"])
    a = ap.parse_args()
    key_f = WORK / "key.json"
    if a.cmd == "build":
        if key_f.exists():
            raise SystemExit(f"{key_f} exists; refusing to reshuffle (scores would no longer match)")
        conds = {t: DATA / c for t, c in CONDITIONS.items()}
        n = {t: len(list(d.glob("*/[0-9][0-9][0-9]/rating-claude.json"))) for t, d in conds.items()}
        if any(v < 420 for v in n.values()):
            raise SystemExit(f"every temperature needs its 420 first-pass ratings first: {n}")
        key = build_key(conds, [p["id"] for p in prompts()])
        WORK.joinpath("scores").mkdir(parents=True, exist_ok=True)
        key_f.write_text(json.dumps(key, indent=1))
        print({pid: len(v) for pid, v in key["prompts"].items()})
    elif a.cmd == "order":  # survey order per prompt for the dispatch message (seeded, independent of the key)
        key = json.loads(key_f.read_text())
        for pid, items in key["prompts"].items():
            ids = [it["blind"] for it in items]
            print(pid, ", ".join(random.Random(f"survey:{pid}").sample(ids, len(ids))))
    else:
        key = json.loads(key_f.read_text())
        df = collect(key, WORK / "scores")
        meta = pd.DataFrame([{"prompt_id": p["id"], "domain": p["domain"], "openness": p["openness"]}
                             for p in prompts()])
        df = df.merge(meta, on="prompt_id", validate="m:1")
        out = ROOT / "results" / "temperature-sweep"
        out.mkdir(parents=True, exist_ok=True)
        df.drop(columns=["blind"]).to_csv(out / "novelty-cross.csv", index=False)
        res = analyse(df)
        res["provenance"] = {
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                         text=True).stdout.strip(),
            "key_seed": key["seed"], "temperatures": key["temperatures"], "decision": "D-041",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        (out / "novelty-cross.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res["effects"]), json.dumps(res["means"]))


if __name__ == "__main__":
    main()
