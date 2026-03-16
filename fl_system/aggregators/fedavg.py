"""fl_system/aggregators/fedavg.py — Federated averaging aggregation."""

from __future__ import annotations

import numpy as np
from typing import List, Optional


def fedavg_aggregate(
    gradient_list: List[np.ndarray],
    weights: Optional[List[int]] = None,
) -> np.ndarray:
    """Weighted average of gradients (FedAvg).

    Args:
        gradient_list: list of gradient vectors, each shape (d,)
        weights:       per-client sample counts; if None, equal weights
                       are used (i.e. plain arithmetic mean)

    Returns:
        Weighted-average gradient of shape (d,)

    Raises:
        ValueError: if gradient_list is empty or weights length mismatches
    """
    if not gradient_list:
        raise ValueError("gradient_list must be non-empty")

    n = len(gradient_list)

    if weights is None:
        weights = [1] * n

    if len(weights) != n:
        raise ValueError(
            f"len(weights)={len(weights)} != len(gradient_list)={n}"
        )

    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("sum(weights) must be > 0")

    aggregated = sum(
        w * g for w, g in zip(weights, gradient_list)
    )
    return aggregated / total_weight
