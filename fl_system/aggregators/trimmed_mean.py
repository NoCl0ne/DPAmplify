"""
fl_system/aggregators/trimmed_mean.py — Trimmed-mean Byzantine-robust aggregation.

Reference:
    Yin, D., Chen, Y., Ramchandran, K., & Bartlett, P. (2018).
    Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates.
    International Conference on Machine Learning (ICML).

Coordinate-wise trimmed mean removes the floor(β·n) largest and
floor(β·n) smallest values at each coordinate before averaging the
remaining values.  This bounds the influence of any β-fraction of
Byzantine workers on the per-coordinate estimate.
"""

from __future__ import annotations

import math
import numpy as np
from typing import List


def trimmed_mean_aggregate(
    gradients: List[np.ndarray],
    beta: float = 0.1,
) -> np.ndarray:
    """Coordinate-wise trimmed mean of gradients.

    At each coordinate d:
        1. Sort the n values g_1[d], …, g_n[d].
        2. Discard the floor(β·n) smallest and floor(β·n) largest.
        3. Return the mean of the remaining n − 2·floor(β·n) values.

    Args:
        gradients: list of gradient vectors, each shape (d,)
        beta:      trimming fraction; must satisfy 0 < beta < 0.5

    Returns:
        Trimmed-mean gradient of shape (d,)

    Raises:
        ValueError: if beta <= 0 or beta >= 0.5, or gradient_list is empty
    """
    if beta <= 0 or beta >= 0.5:
        raise ValueError(
            f"beta must be in (0, 0.5), got {beta}"
        )
    if not gradients:
        raise ValueError("gradients list must be non-empty")

    n = len(gradients)
    trim = math.floor(beta * n)

    # Stack into matrix (n, d)
    G = np.stack(gradients, axis=0)  # shape (n, d)

    # Sort each column
    G_sorted = np.sort(G, axis=0)

    # Trim top and bottom `trim` rows
    if trim > 0:
        G_trimmed = G_sorted[trim : n - trim, :]
    else:
        G_trimmed = G_sorted

    return G_trimmed.mean(axis=0)
