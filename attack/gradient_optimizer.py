"""
attack/gradient_optimizer.py — Constructs adversarial gradients for DPAmplify.

Core construction:
    g_adv = C · (g_target / ||g_target||₂)

Since ||g_adv||₂ = C exactly:
    clip(g_adv, C) = g_adv            (clip is the identity)
    E[M_DP(g_adv)] = g_adv            (noise has zero mean)

The adversarial gradient is therefore aligned with g_target and
survives the DP mechanism with zero bias.
"""

from __future__ import annotations

import numpy as np

from theory.snr_analysis import (
    compute_attack_snr_upper_bound,
    compute_attack_snr_tight,
)


class GradientOptimizer:
    """Constructs and analyses adversarial gradients for DPAmplify.

    Attributes:
        C (float): clipping threshold used to scale g_adv
    """

    def __init__(self, g_target: np.ndarray, C: float) -> None:
        """
        Args:
            g_target: target direction (any non-zero vector); will be
                      normalised internally to a unit vector
            C:        clipping threshold

        Raises:
            ValueError: if ||g_target||₂ < 1e-10 (direction undefined)
        """
        norm = np.linalg.norm(g_target)
        if norm < 1e-10:
            raise ValueError(
                f"||g_target||₂ = {norm:.2e} < 1e-10; "
                "target direction is undefined (zero vector)"
            )
        self._g_target_unit: np.ndarray = g_target / norm
        self.C = float(C)

    # ── Adversarial gradient ─────────────────────────────────────────

    def compute_g_adv(self) -> np.ndarray:
        """Return g_adv = C · (g_target / ||g_target||₂).

        By construction ||g_adv||₂ = C exactly, so the DP clipping
        operator is the identity on g_adv:

            clip(g_adv, C) = g_adv
            E[M_DP(g_adv)] = g_adv    (since E[ξ] = 0)

        The gradient is aligned with g_target and preserved in
        expectation through any Gaussian DP mechanism with threshold C.

        Returns:
            Adversarial gradient vector of same shape as g_target.
        """
        return self.C * self._g_target_unit

    # ── Expected aggregate contribution ─────────────────────────────

    def compute_expected_contribution(self, k: int, n: int) -> np.ndarray:
        """Expected Byzantine contribution to the FedAvg aggregate.

        In a federation of n clients with k Byzantine:
            E[k · M_DP(g_adv) / n] = k/n · g_adv

        Args:
            k: number of Byzantine clients
            n: total number of clients

        Returns:
            (k / n) · g_adv
        """
        return (k / n) * self.compute_g_adv()

    # ── SNR analysis ─────────────────────────────────────────────────

    def compute_snr_upper(self, k: int, n: int, sigma: float) -> float:
        """SNR upper bound (Theorem 1a): k·C / (σ·√(n−k)).

        Args:
            k:     number of Byzantine clients
            n:     total number of clients
            sigma: DP noise standard deviation

        Returns:
            Upper-bound SNR (float)
        """
        return compute_attack_snr_upper_bound(k, n, self.C, sigma)

    def compute_snr_tight(
        self, k: int, n: int, sigma: float, var_honest: float
    ) -> float:
        """Tight SNR (Theorem 1b): signal / sqrt(σ²/n + (n−k)·Var_h/n²).

        Args:
            k:          number of Byzantine clients
            n:          total number of clients
            sigma:      DP noise standard deviation
            var_honest: Var[ clip(g_h, C) · g_target ] per honest client

        Returns:
            Tight SNR estimate (float)
        """
        return compute_attack_snr_tight(k, n, self.C, sigma, var_honest)

    # ── Empirical verification ───────────────────────────────────────

    def verify_no_clipping(
        self,
        mechanism: object,
        n_samples: int = 1000,
        rng: np.random.Generator = None,
    ) -> bool:
        """Verify empirically that g_adv is not attenuated by the DP mechanism.

        Draws n_samples from M_DP(g_adv) and checks that the sample mean
        is within 5 % of C from g_adv in L2 distance.

        Args:
            mechanism: DPMechanism instance (must have sample_outputs method
                       and a .C attribute matching self.C)
            n_samples: number of mechanism draws for the Monte-Carlo estimate
            rng:       numpy random generator

        Returns:
            True if ||E[M_DP(g_adv)] - g_adv||₂ < 0.05 · C
        """
        if rng is None:
            rng = np.random.default_rng()
        g_adv = self.compute_g_adv()
        samples = mechanism.sample_outputs(g_adv, n_samples, rng)
        mean_output = samples.mean(axis=0)
        error = np.linalg.norm(mean_output - g_adv)
        return bool(error < 0.05 * mechanism.C)
