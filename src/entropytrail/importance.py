"""Fork importance (spec §5.4, D-035).

importance = mean outcome distance between alternative-token and same-token rollouts
             − mean outcome distance among the same-token rollouts (noise floor).
``importance_incl_original`` (secondary) adds the parent app as a third same-token outcome.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from entropytrail.outcome import Outcome, outcome_distance


def _parts(alts: list[Outcome], sames: list[Outcome]) -> tuple[float, float, float] | None:
    if len(alts) < 1 or len(sames) < 2:
        return None
    alt_same = float(np.mean([outcome_distance(a, s) for a in alts for s in sames]))
    floor = float(np.mean([outcome_distance(a, b) for a, b in combinations(sames, 2)]))
    return alt_same, floor, alt_same - floor


def fork_importance(alts: list[Outcome], sames: list[Outcome], original: Outcome | None = None) -> dict:
    p = _parts(alts, sames)
    q = _parts(alts, sames + [original]) if original is not None else None
    return {
        "alt_same": p[0] if p else None, "noise_floor": p[1] if p else None,
        "importance": p[2] if p else None,
        "importance_incl_original": q[2] if q else None,
        "n_alt": len(alts), "n_same": len(sames),
    }
