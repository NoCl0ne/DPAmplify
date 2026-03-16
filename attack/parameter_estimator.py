"""
attack/parameter_estimator.py — Passive estimation of DP parameters.

During the estimation phase (first T_est rounds), the Byzantine client
behaves honestly and records the L2 norms of gradient updates it sends
to the server.  After accumulating enough observations it estimates:

    C̃ ≈ percentile(observed_norms, p)    (default p = 90)
    σ̃ ≈ std(observed_norms)

Rationale: clipped gradients are bounded by C, so the p-th percentile
of observed norms approaches C from below.  The spread in norms is
dominated by the DP noise scale σ.
"""

from __future__ import annotations

import numpy as np
from typing import List


class PassiveParameterEstimator:
    """Estimates the DP clipping threshold C and noise σ from observed norms.

    The Byzantine client feeds observed gradient norms (its own, or
    norms derived from server-side messages) into this estimator during
    the honest estimation phase.

    Attributes:
        history_window (int): minimum observations required before estimating
        percentile_C   (float): percentile of norms used to estimate C
    """

    def __init__(
        self,
        history_window: int = 20,
        percentile_C: float = 90.0,
    ) -> None:
        """
        Args:
            history_window: number of observations required before
                            is_ready() returns True
            percentile_C:   percentile (0–100) of observed norms
                            used to estimate C
        """
        self.history_window = history_window
        self.percentile_C = percentile_C
        self._norm_history: List[float] = []

    # ── Core observation ─────────────────────────────────────────────

    def update(self, observed_norm: float) -> None:
        """Record a gradient norm observation.

        Args:
            observed_norm: L2 norm of an observed gradient update
        """
        self._norm_history.append(float(observed_norm))

    # ── Readiness check ──────────────────────────────────────────────

    def is_ready(self) -> bool:
        """True when at least history_window observations have been recorded."""
        return len(self._norm_history) >= self.history_window

    # ── Estimates ────────────────────────────────────────────────────

    def estimate_C(self) -> float:
        """Estimate the clipping threshold as the p-th percentile of norms.

        Returns:
            Estimated C (float)

        Raises:
            RuntimeError: if is_ready() is False
        """
        self._require_ready()
        return float(np.percentile(self._norm_history, self.percentile_C))

    def estimate_sigma(self) -> float:
        """Estimate the DP noise scale as the standard deviation of norms.

        Returns:
            Estimated σ (float)

        Raises:
            RuntimeError: if is_ready() is False
        """
        self._require_ready()
        return float(np.std(self._norm_history))

    def get_estimates(self) -> dict:
        """Return all current estimates as a dictionary.

        Returns:
            {"C": float, "sigma": float, "n_observations": int}

        Raises:
            RuntimeError: if is_ready() is False
        """
        self._require_ready()
        return {
            "C": self.estimate_C(),
            "sigma": self.estimate_sigma(),
            "n_observations": len(self._norm_history),
        }

    def reset(self) -> None:
        """Clear all recorded observations."""
        self._norm_history = []

    # ── Internal ─────────────────────────────────────────────────────

    def _require_ready(self) -> None:
        if not self.is_ready():
            raise RuntimeError(
                f"Not enough observations: {len(self._norm_history)} "
                f"< {self.history_window}"
            )


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    true_C = 1.0
    true_sigma = 0.1

    est = PassiveParameterEstimator(history_window=20)
    for _ in range(25):
        # Simulate norms of clipped + noisy gradients ~ N(C, σ²)
        norm = rng.normal(true_C, true_sigma)
        est.update(max(norm, 0.0))

    estimates = est.get_estimates()
    print(f"True       C = {true_C:.3f},  sigma = {true_sigma:.3f}")
    print(
        f"Estimated  C = {estimates['C']:.3f},  "
        f"sigma = {estimates['sigma']:.3f}"
    )
    print(f"Observations : {estimates['n_observations']}")
