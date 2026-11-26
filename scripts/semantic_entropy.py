"""Semantic entropy from the D-041 idea lines (D-042, exploratory).

    uv run python scripts/semantic_entropy.py lines <prompt_id>   # idea lines for the cluster agent
    uv run --group analysis python scripts/semantic_entropy.py collect

`lines` prints the 60 idea lines of a prompt under their blind ids in a seeded order (no
temperatures). One agent per prompt (docs/semantic-cluster-brief.md) writes
/Volumes/Data/entropytrail/_novelty-cross/clusters/<prompt_id>.json. `collect` writes
results/temperature-sweep/semantic-entropy.csv and semantic-entropy.json.
"""
import argparse
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path

from entropytrail.analysis.semantic import analyse, semantic_table

ROOT = Path(__file__).resolve().parents[1]
WORK = Path("/Volumes/Data/entropytrail/_novelty-cross")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["lines", "collect"])
    ap.add_argument("prompt_id", nargs="?")
    a = ap.parse_args()
    key = json.loads((WORK / "key.json").read_text())
    if a.cmd == "lines":
        scores = json.loads((WORK / "scores" / f"{a.prompt_id}.json").read_text())["scores"]
        ids = sorted(scores)
        for b in random.Random(f"cluster:{a.prompt_id}").sample(ids, len(ids)):
            print(f"{b}: {scores[b].get('idea', '').strip()}")
        return
    tab = semantic_table(key, WORK / "clusters")
    out = ROOT / "results" / "temperature-sweep"
    tab.to_csv(out / "semantic-entropy.csv", index=False)
    res = analyse(tab)
    res["provenance"] = {
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                     text=True).stdout.strip(),
        "decision": "D-042", "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (out / "semantic-entropy.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["effects"]), json.dumps(res["means"]))


if __name__ == "__main__":
    main()
