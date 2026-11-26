"""Build the human rating site (D-043) from the fixed rating sample.

    uv run python scripts/build_rating_site.py

Reads results/rating-sample.csv (105 apps) and prompts/prompts.yaml, gives each app a blind
id (a001 ... in a seeded random order, so ids reveal neither prompt nor sample), and writes

    worker/public/apps/<id>.html the app, copied from the raw data on the Studio; served by the
                                 ratings Worker, i.e. from another origin than the rating page
    site/apps.json               [{id, prompt}] for the rating page
    worker/src/apps.json         the id list for the ratings Worker
    results/rating-sample-ids.csv  id -> prompt_id, sample (used when the ratings are analysed)

Refuses to change existing ids: if rating-sample-ids.csv exists, its mapping is reused.

Anchor apps (D-044): three grid apps outside the sample, rated first by every rater so that
rater scale use can be estimated. Fixed by an automatic rule on the non-sample T = 1.0 apps:
k1 = does not render and has the emptiest load screenshot (grayscale image entropy, ties by
prompt id and sample); k2 = renders without any page or console error and has the fewest
characters; k3 = renders without any error and has the highest screenshot entropy.
"""
import csv
import json
import random
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/Volumes/Data/entropytrail/qwen35-9b-bf16-mlx-1195d88b")
SEED = 43
ANCHORS = [("k1", "free-02", 2), ("k2", "tool-tight-01", 10), ("k3", "free-03", 18)]


def assign_ids(sample: list[dict], seed: int = SEED) -> list[dict]:
    items = sorted(sample, key=lambda r: (r["prompt_id"], int(r["sample"])))
    random.Random(f"ids:{seed}").shuffle(items)
    return [{"id": f"a{i + 1:03d}", "prompt_id": r["prompt_id"], "sample": int(r["sample"])}
            for i, r in enumerate(items)]


def main() -> None:
    ids_f = ROOT / "results" / "rating-sample-ids.csv"
    if ids_f.exists():
        mapping = [dict(r, sample=int(r["sample"])) for r in csv.DictReader(ids_f.open())]
    else:
        mapping = assign_ids(list(csv.DictReader((ROOT / "results" / "rating-sample.csv").open())))
        with ids_f.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "prompt_id", "sample"])
            w.writeheader()
            w.writerows(mapping)
    prompts = {p["id"]: p["text"].strip() for p in yaml.safe_load((ROOT / "prompts" / "prompts.yaml").read_text())["prompts"]}
    apps_dir = ROOT / "worker" / "public" / "apps"
    apps_dir.mkdir(parents=True, exist_ok=True)
    for m in mapping:
        shutil.copyfile(DATA / m["prompt_id"] / f"{m['sample']:03d}" / "app.html", apps_dir / f"{m['id']}.html")
    anchors = [{"id": a, "prompt_id": p, "sample": s} for a, p, s in ANCHORS]
    for m in anchors:
        shutil.copyfile(DATA / m["prompt_id"] / f"{m['sample']:03d}" / "app.html", apps_dir / f"{m['id']}.html")
    with (ROOT / "results" / "rating-anchors.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "prompt_id", "sample"])
        w.writeheader()
        w.writerows(anchors)
    (ROOT / "site" / "apps.json").write_text(json.dumps(
        [{"id": m["id"], "prompt": prompts[m["prompt_id"]]} for m in sorted(mapping + anchors, key=lambda m: m["id"])],
        indent=1, ensure_ascii=False))
    (ROOT / "worker" / "src").mkdir(parents=True, exist_ok=True)
    (ROOT / "worker" / "src" / "apps.json").write_text(json.dumps(sorted(m["id"] for m in mapping)))
    (ROOT / "worker" / "src" / "anchors.json").write_text(json.dumps([a for a, _, _ in ANCHORS]))
    print(len(mapping), "apps")


if __name__ == "__main__":
    main()
