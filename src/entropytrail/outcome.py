"""Outcome distance between apps and ensemble spread / rarity (WP4, D-034).

An app's screen embedding is the L2-normalised mean of the image embeddings of its two
screenshots (after load, after the fixed interaction). Outcome distance (spec §5.4) is 1.0
when exactly one of two apps renders, else the screen cosine distance clipped to [0, 1].
Ensemble spread and rarity (frozen plan) use the screen cosine distance without the
render rule unless ``use_render_rule`` is set.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Outcome:
    renders: bool
    screen: np.ndarray


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def screen_embedding(img_load: np.ndarray, img_interact: np.ndarray) -> np.ndarray:
    return _unit(np.asarray(img_load, np.float32) + np.asarray(img_interact, np.float32))


def outcome_distance(a: Outcome, b: Outcome) -> float:
    if a.renders != b.renders:
        return 1.0
    return float(np.clip(1.0 - float(a.screen @ b.screen), 0.0, 1.0))


def cosine_distances(emb: np.ndarray) -> np.ndarray:
    E = np.asarray(emb, np.float64)
    D = np.clip(1.0 - E @ E.T, 0.0, 1.0)
    np.fill_diagonal(D, 0.0)
    return D


def outcome_distances(outcomes: list[Outcome]) -> np.ndarray:
    D = cosine_distances(np.stack([o.screen for o in outcomes]))
    r = np.array([o.renders for o in outcomes])
    D[r[:, None] != r[None, :]] = 1.0
    return D


def spread(D: np.ndarray) -> float:
    n = D.shape[0]
    if n < 2:
        return float("nan")
    return float(D[np.triu_indices(n, 1)].mean())


def rarity(D: np.ndarray) -> np.ndarray:
    n = D.shape[0]
    if n < 2:
        return np.full(n, np.nan)
    return D.sum(axis=1) / (n - 1)


def _screen_matrix(outcomes: list[Outcome], use_render_rule: bool) -> np.ndarray:
    if use_render_rule:
        return outcome_distances(outcomes)
    return cosine_distances(np.stack([o.screen for o in outcomes]))


def ensemble_spread(outcomes: list[Outcome], use_render_rule: bool = False) -> float:
    return spread(_screen_matrix(outcomes, use_render_rule))


def ensemble_rarity(outcomes: list[Outcome], use_render_rule: bool = False) -> np.ndarray:
    return rarity(_screen_matrix(outcomes, use_render_rule))
