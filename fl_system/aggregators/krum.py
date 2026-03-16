"""
fl_system/aggregators/krum.py — Krum Byzantine-robust aggregation.

Reference:
    Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017).
    Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent.
    Advances in Neural Information Processing Systems (NeurIPS), 30.

Krum selects the gradient whose sum of squared distances to its
(n − f − 2) nearest neighbours is smallest, where f is the assumed
number of Byzantine clients.  This selection is robust to f < n/2
Byzantine workers.

Worked example (n=5, f=1):
    For each gradient g_i, compute the sum of squared L2 distances
    to its (5 − 1 − 2) = 2 nearest neighbours among the other 4.
    Select the g_i with the smallest such sum.
"""

from __future__ import annotations

import numpy as np
from typing import List


def krum_aggregate(gradients: List[np.ndarray], f: int) -> np.ndarray:
    """Select the gradient least influenced by Byzantine workers (Krum).

    For each gradient g_i, computes the score:
        s(i) = sum of squared distances to the (n − f − 2) nearest
               neighbours of g_i among {g_j : j ≠ i}
    Returns the g_i with the minimum score.

    Args:
        gradients: list of gradient vectors, each shape (d,)
        f:         number of assumed Byzantine clients

    Returns:
        The selected gradient (shape (d,))

    Raises:
        ValueError: if f >= len(gradients) // 2, which violates the
                    Krum assumption of a strict minority of Byzantines
    """
    n = len(gradients)
    if f >= n // 2:
        raise ValueError(
            f"Krum requires f < n/2, but f={f} >= n//2={n // 2}. "
            "Byzantine clients must be a strict minority."
        )

    # n_neighbours: how many nearest neighbours to sum over
    n_neighbours = n - f - 2
    if n_neighbours < 1:
        raise ValueError(
            f"n − f − 2 = {n_neighbours} < 1; too few honest clients "
            f"(n={n}, f={f})"
        )

    # Pairwise squared distances
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diff = gradients[i] - gradients[j]
            sq_dist = float(np.dot(diff, diff))
            distances[i, j] = sq_dist
            distances[j, i] = sq_dist

    # Krum score for each gradient
    scores = np.zeros(n)
    for i in range(n):
        row = distances[i].copy()
        row[i] = np.inf  # exclude self
        nearest = np.sort(row)[:n_neighbours]
        scores[i] = nearest.sum()

    best_idx = int(np.argmin(scores))
    return gradients[best_idx].copy()
