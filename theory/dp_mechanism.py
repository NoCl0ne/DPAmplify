"""
theory/dp_mechanism.py — Gaussian DP mechanism for DPAmplify.

The Gaussian DP mechanism is:

    M_DP(g) = clip(g, C) + ξ,   ξ ~ N(0, σ²I)

    clip(g, C) = g · min(1, C / max(||g||₂, ε))

Since E[ξ] = 0, the expectation is:

    E[M_DP(g)] = clip(g, C)

Key property exploited by DPAmplify:
    If ||g||₂ = C exactly, clip is the identity map, so
    E[M_DP(g)] = g — the gradient survives the DP mechanism
    without any attenuation in expectation.
"""

import numpy as np


class DPMechanism:
    """Gaussian DP mechanism: clip gradient to norm C, then add N(0, σ²I) noise.

    Attributes:
        C (float): clipping threshold (L2 norm bound)
        sigma (float): noise standard deviation
    """

    def __init__(self, C: float, sigma: float) -> None:
        """
        Args:
            C:     clipping threshold; must be > 0
            sigma: noise standard deviation; must be > 0

        Raises:
            ValueError: if C <= 0 or sigma <= 0
        """
        if C <= 0:
            raise ValueError(f"C must be > 0, got {C}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.C = C
        self.sigma = sigma

    def clip(self, g: np.ndarray) -> np.ndarray:
        """Clip gradient to L2 norm C.

        Formula:
            clip(g, C) = g · min(1, C / max(||g||₂, ε))

        If ||g||₂ ≤ C, g is returned unchanged.
        Otherwise g is scaled down to have norm exactly C.

        Args:
            g: gradient vector of any shape

        Returns:
            Clipped gradient, same shape as g.
        """
        norm = np.linalg.norm(g)
        if norm <= self.C:
            return g.copy()
        return g * (self.C / max(norm, 1e-10))

    def add_noise(
        self, g: np.ndarray, rng: np.random.Generator = None
    ) -> np.ndarray:
        """Add Gaussian noise N(0, σ²I) to gradient.

        Args:
            g:   gradient vector
            rng: numpy random generator; creates a new one if None

        Returns:
            g + ξ where ξ ~ N(0, σ²I), same shape as g.
        """
        if rng is None:
            rng = np.random.default_rng()
        return g + rng.normal(0.0, self.sigma, size=g.shape)

    def apply(
        self, g: np.ndarray, rng: np.random.Generator = None
    ) -> np.ndarray:
        """Apply the full DP mechanism: M_DP(g) = clip(g, C) + N(0, σ²I).

        Args:
            g:   gradient vector
            rng: numpy random generator; creates a new one if None

        Returns:
            clip(g, C) + ξ where ξ ~ N(0, σ²I).
        """
        return self.add_noise(self.clip(g), rng)

    def expected_output(self, g: np.ndarray) -> np.ndarray:
        """Return E[M_DP(g)] = clip(g, C).

        Since E[ξ] = 0, the expectation of the DP mechanism equals the
        clipped gradient. For adversarial gradients with ||g||₂ = C exactly,
        clip is the identity, so E[M_DP(g)] = g.

        Args:
            g: gradient vector

        Returns:
            clip(g, C), same shape as g.
        """
        return self.clip(g)

    def sample_outputs(
        self,
        g: np.ndarray,
        n_samples: int,
        rng: np.random.Generator = None,
    ) -> np.ndarray:
        """Draw n_samples independent outputs of M_DP(g).

        Clips once and then draws n_samples independent noise vectors,
        which is equivalent to n_samples independent mechanism applications.

        Args:
            g:        gradient vector of shape (d,)
            n_samples: number of independent draws
            rng:      numpy random generator; creates a new one if None

        Returns:
            Array of shape (n_samples, d) with independent M_DP(g) draws.
        """
        if rng is None:
            rng = np.random.default_rng()
        clipped = self.clip(g)
        noise = rng.normal(0.0, self.sigma, size=(n_samples, g.shape[0]))
        return clipped[np.newaxis, :] + noise


if __name__ == "__main__":
    mech = DPMechanism(C=1.0, sigma=0.1)
    rng = np.random.default_rng(42)
    g = np.array([2.0, 0.0, 0.0])

    clipped = mech.clip(g)
    expected = mech.expected_output(g)
    samples = mech.sample_outputs(g, 1000, rng)
    sample_mean = samples.mean(axis=0)

    print(f"g                   = {g}")
    print(f"clip(g, C=1.0)      = {clipped}")
    print(f"E[M_DP(g)]          = {expected}")
    print(f"sample mean (n=1000)= {sample_mean.round(4)}")

    l2_error = np.linalg.norm(sample_mean - expected)
    assert l2_error < 0.05, f"Self-test FAILED: L2 error {l2_error:.4f} >= 0.05"
    print("DPMechanism self-test passed.")
