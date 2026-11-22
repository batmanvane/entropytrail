"""Collect the eval/ folders of the grid apps into results/outcomes.csv and results/ensemble.csv.

    uv run python scripts/collect_outcomes.py [--root /Volumes/Data/entropytrail/<condition_id>]

Columns are documented in docs/DATA.md. Rarity and spread use the screen embedding without
the render rule; `ensemble_spread_render_rule` uses the outcome distance (D-034).
Every done app must have a complete eval/ folder (run `et eval` first).
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from entropytrail.evaluate import find_units
from entropytrail.fingerprint import fingerprint_distance
from entropytrail.outcome import Outcome, cosine_distances, outcome_distances, rarity, spread

ROOT = Path(__file__).resolve().parents[1]
APP_FIELDS = ["prompt_id", "sample", "domain", "openness", "n_tokens", "finish_reason", "html_extracted",
              "n_chars", "n_lines", "renders", "timeout", "n_page_errors", "n_console_errors",
              "n_page_errors_after_interaction", "ensemble_rarity", "code_rarity", "fingerprint_rarity"]
ENS_FIELDS = ["prompt_id", "domain", "openness", "n_apps", "n_renders", "ensemble_spread",
              "ensemble_spread_render_rule", "code_spread", "fingerprint_spread"]
RENDER_KEYS = ["renders", "timeout", "n_page_errors", "n_console_errors", "n_page_errors_after_interaction"]


def _load(u: Path) -> dict:
    ev = u / "eval"
    if not (ev / "eval.json").exists():
        raise FileNotFoundError(f"{u} has no eval/ (run et eval first)")
    meta = json.loads((u / "meta.json").read_text())
    emb = np.load(ev / "embeddings.npz")
    app = u / "app.html"
    html = app.read_text(encoding="utf-8") if app.exists() else ""
    return {
        "meta": meta, "render": json.loads((ev / "render.json").read_text()),
        "fp": json.loads((ev / "fingerprint.json").read_text()),
        "screen": emb["screen"], "code": emb["code"],
        "n_chars": len(html), "n_lines": html.count("\n") + 1 if html else 0,
    }


def collect(cond_dir: Path) -> tuple[list[dict], list[dict]]:
    by_prompt: dict[str, list[tuple[Path, dict]]] = {}
    for u in find_units(cond_dir, prompt_ids=None, forks=False):
        by_prompt.setdefault(u.parent.name, []).append((u, _load(u)))
    apps, ens = [], []
    for pid, items in sorted(by_prompt.items()):
        data = [d for _, d in items]
        outs = [Outcome(bool(d["render"]["renders"]), d["screen"]) for d in data]
        D_screen = cosine_distances(np.stack([o.screen for o in outs]))
        D_code = cosine_distances(np.stack([d["code"] for d in data]))
        n = len(data)
        D_fp = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D_fp[i, j] = D_fp[j, i] = fingerprint_distance(data[i]["fp"], data[j]["fp"])
        r_screen, r_code, r_fp = rarity(D_screen), rarity(D_code), rarity(D_fp)
        prompt = data[0]["meta"].get("prompt", {})
        for k, ((u, d), rs, rc, rf) in enumerate(zip(items, r_screen, r_code, r_fp)):
            m = d["meta"]
            apps.append({
                "prompt_id": pid, "sample": u.name, "domain": prompt.get("domain"),
                "openness": prompt.get("openness"), "n_tokens": m.get("n_tokens"),
                "finish_reason": m.get("finish_reason"), "html_extracted": m.get("html_extracted"),
                "n_chars": d["n_chars"], "n_lines": d["n_lines"],
                **{key: d["render"].get(key) for key in RENDER_KEYS},
                "ensemble_rarity": float(rs), "code_rarity": float(rc), "fingerprint_rarity": float(rf),
            })
        ens.append({
            "prompt_id": pid, "domain": prompt.get("domain"), "openness": prompt.get("openness"),
            "n_apps": n, "n_renders": sum(o.renders for o in outs),
            "ensemble_spread": spread(D_screen), "ensemble_spread_render_rule": spread(outcome_distances(outs)),
            "code_spread": spread(D_code), "fingerprint_spread": spread(D_fp),
        })
    return apps, ens


def _write(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Volumes/Data/entropytrail/qwen35-9b-bf16-mlx-1195d88b", type=Path)
    ap.add_argument("--out", default=ROOT / "results", type=Path)
    a = ap.parse_args()
    apps, ens = collect(a.root)
    _write(a.out / "outcomes.csv", APP_FIELDS, apps)
    _write(a.out / "ensemble.csv", ENS_FIELDS, ens)
    print(f"{len(apps)} apps, {len(ens)} prompts -> {a.out}")


if __name__ == "__main__":
    main()
