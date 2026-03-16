"""
countermeasures/gradient_auditor.py — Norm-based detection of DPAmplify.

Defense rationale:
    DPAmplify always submits gradients with ||g_adv||₂ = C exactly.
    In standard DP-FL, honest gradients span a range of norms below C;
    a Byzantine client always contributing at exactly C is statistically
    anomalous.

    Two detectors are provided:
      1. audit_gradient_norms — flags individual gradients too close to C.
      2. norm_spike_detector  — flags a client whose norm history has
                                unusually low variance (constant-norm signal).

Caveat: both detectors have false-positive costs under low-noise regimes
where honest gradients also concentrate near C.  See the paper's
Section V for a full precision-recall analysis.
"""

from __future__ import annotations

import numpy as np
from typing import List


def audit_gradient_norms(
    norms: List[float],
    C_estimated: float,
    tolerance: float = 0.01,
) -> List[bool]:
    """Flag gradients whose norm is suspiciously close to C.

    A gradient is flagged if |||g||₂ − Ĉ| < tolerance, indicating it
    was constructed to sit exactly at the clipping boundary — the
    hallmark of DPAmplify's g_adv = C · ê construction.

    Args:
        norms:       list of per-client gradient L2 norms for one round
        C_estimated: estimated clipping threshold Ĉ
        tolerance:   absolute norm distance from Ĉ to trigger a flag

    Returns:
        List of bool, True for each gradient that is flagged as suspicious.
    """
    return [
        abs(norm - C_estimated) < tolerance
        for norm in norms
    ]


def norm_spike_detector(
    norm_history: List[float],
    window: int = 10,
) -> bool:
    """Detect a client that consistently submits near-constant norms.

    A client is flagged if the standard deviation of its last `window`
    gradient norms is less than 2 % of their mean.  Honest clients
    exhibit natural norm variation due to data heterogeneity and DP noise;
    a Byzantine client running DPAmplify always submits ||g||₂ ≈ C,
    producing anomalously low variance.

    Args:
        norm_history: sequence of gradient norms over consecutive rounds
                      (most recent norm is last)
        window:       number of recent rounds to inspect

    Returns:
        True if the client's norm history appears suspiciously stable.
    """
    if len(norm_history) < window:
        return False

    recent = norm_history[-window:]
    mean   = float(np.mean(recent))
    std    = float(np.std(recent))

    if mean < 1e-10:
        return False

    return (std / mean) < 0.02
