"""Semantic entropy of the app ensembles across temperatures (D-042, exploratory).

The idea lines written in the blind cross-temperature novelty pass (D-041) are clustered per
prompt into "same core idea" classes by one agent per prompt that sees only the 60 idea lines
under blind ids (``docs/semantic-cluster-brief.md``). Semantic entropy of a prompt at a
temperature is the entropy (bits) of the class distribution of its 20 apps (Kuhn, Gal &
Farquhar 2023: sample, cluster by meaning, entropy over clusters). Because the classes are
formed once over the pooled apps, the temperatures share one partition.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd

LABEL = ("exploratory (D-042) - semantic entropy over Claude-clustered idea lines of the D-041 pool; "
         "defined after the frozen analyses; not part of any Holm family")


def entropy_bits(counts: list[int]) -> tuple[float, float]:
    """Plug-in entropy and Miller-Madow corrected entropy (bits)."""
    n = sum(counts)
    c = [k for k in counts if k > 0]
    if n == 0:
        return math.nan, math.nan
    h = -sum(k / n * math.log2(k / n) for k in c)
    return h, h + (len(c) - 1) / (2 * n * math.log(2))


def load_partition(path: Path, ids: set[str]) -> dict[str, int]:
    """{blind id: cluster index}; every id exactly once."""
    data = json.loads(Path(path).read_text())
    seen: dict[str, int] = {}
    for i, cl in enumerate(data["clusters"]):
        for b in cl["ids"]:
            if b in seen:
                raise ValueError(f"{path}: id {b} in two clusters")
            seen[b] = i
    if set(seen) != ids:
        raise ValueError(f"{path}: ids missing {sorted(ids - set(seen))} or unknown {sorted(set(seen) - ids)}")
    return seen


def semantic_table(key: dict, clusters_dir: Path) -> pd.DataFrame:
    rows = []
    for pid, items in key["prompts"].items():
        part = load_partition(Path(clusters_dir) / f"{pid}.json", {it["blind"] for it in items})
        n_pool = len(set(part.values()))
        by_t: dict[float, Counter] = {}
        for it in items:
            by_t.setdefault(it["temperature"], Counter())[part[it["blind"]]] += 1
        for t, cnt in by_t.items():
            h, hmm = entropy_bits(list(cnt.values()))
            rows.append({"prompt_id": pid, "temperature": t, "n_apps": sum(cnt.values()),
                         "n_clusters": len(cnt), "n_clusters_pool": n_pool,
                         "top_cluster_share": max(cnt.values()) / sum(cnt.values()),
                         "H_sem": h, "H_sem_mm": hmm})
    return pd.DataFrame(rows)


def analyse(tab: pd.DataFrame, ref: float = 1.0) -> dict:
    from scipy.stats import wilcoxon

    wide = tab.pivot(index="prompt_id", columns="temperature", values="H_sem")
    effects = {}
    for t in sorted((c for c in wide.columns if c != ref), reverse=True):
        d = (wide[t] - wide[ref]).dropna()
        nz = d[d != 0]
        p = float(wilcoxon(nz, alternative="less").pvalue) if len(nz) >= 5 else math.nan
        effects[str(t)] = {"mean_diff_bits": float(d.mean()), "median_diff_bits": float(d.median()),
                           "n_prompts": len(d), "n_lower": int((d < 0).sum()), "n_higher": int((d > 0).sum()),
                           "wilcoxon_p_one_sided_lower": p}
    means = tab.groupby("temperature")[["H_sem", "H_sem_mm", "n_clusters", "top_cluster_share"]].mean()
    return {"label": LABEL, "effects": effects,
            "means": {str(t): {k: float(v) for k, v in r.items()} for t, r in means.iterrows()},
            "test": "paired Wilcoxon signed-rank over prompts, H_sem(T) - H_sem(1.0), one-sided lower"}
