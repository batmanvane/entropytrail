"""Stopping and entropy over position across temperatures (D-042, exploratory).

    uv run --group analysis python scripts/stopping_analysis.py

Needs `et rescore` for every grid unit of the three conditions (queue jobs `rescore …`).
Writes results/temperature-sweep/stopping-per-app.csv, stopping-curves.csv and stopping.json.
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from entropytrail.analysis.stopping import summarise, unit_stop
from entropytrail.rescore import grid_units, is_rescored

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/Volumes/Data/entropytrail")
CONDITIONS = {1.0: "qwen35-9b-bf16-mlx-1195d88b", 0.7: "qwen35-9b-bf16-mlx-t0p7-a31fc218",
              0.5: "qwen35-9b-bf16-mlx-t0p5-c9bce32f"}


def main() -> int:
    apps, curves = [], []
    for t, cid in CONDITIONS.items():
        units = grid_units(DATA / cid)
        missing = [u for u in units if not is_rescored(u)]
        if len(units) != 420 or missing:
            print(f"{cid}: {len(units)} units, {len(missing)} not rescored", file=sys.stderr)
            return 1
        for u in units:
            row, cur = unit_stop(u)
            meta = json.loads((u / "meta.json").read_text())
            apps.append(dict(row, temperature=t, condition_id=cid, prompt_id=u.parent.name, sample=int(u.name),
                             finish_reason=meta["finish_reason"]))
            curves.append(cur.assign(temperature=t, prompt_id=u.parent.name, sample=int(u.name)))
    apps, curves = pd.DataFrame(apps), pd.concat(curves, ignore_index=True)
    out = ROOT / "results" / "temperature-sweep"
    apps.to_csv(out / "stopping-per-app.csv", index=False)
    res = summarise(apps, curves)
    pd.DataFrame(res.pop("curves")).to_csv(out / "stopping-curves.csv", index=False)
    res["provenance"] = {
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                     text=True).stdout.strip(),
        "decision": "D-042", "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (out / "stopping.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["per_temperature"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
