"""
countermeasures/randomized_clipping.py — Proposed defense against DPAmplify.

Defense rationale:
    DPAmplify requires knowing C to construct g_adv = C · ê.
    If the server instead samples C_t ~ U(C_min, C_max) fresh each round,
    the passive estimator cannot converge to a stable value: any estimate
    incurs an irreducible error of ~ (C_max - C_min) / (2√3), the std of
    the uniform distribution.

    The Byzantine client's SNR degrades because its gradient norm no longer
    equals the current clipping threshold, reintroducing attenuation:
        clip(g_adv, C_t) = g_adv * min(1, C_t / ||g_adv||)
    which is no longer the identity when C_t ≠ ||g_adv||.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from attack.parameter_estimator import PassiveParameterEstimator


def randomized_clip(
    g: np.ndarray,
    C_min: float,
    C_max: float,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, float]:
    """Clip gradient to a uniformly random threshold C_t ~ U(C_min, C_max).

    Args:
        g:     gradient vector
        C_min: minimum clipping threshold (must be > 0)
        C_max: maximum clipping threshold (must be > C_min)
        rng:   numpy random generator; creates a new one if None

    Returns:
        (clipped_gradient, C_t)  where C_t is the sampled threshold

    Raises:
        ValueError: if C_min <= 0 or C_max <= C_min
    """
    if C_min <= 0:
        raise ValueError(f"C_min must be > 0, got {C_min}")
    if C_max <= C_min:
        raise ValueError(f"C_max must be > C_min, got C_max={C_max}, C_min={C_min}")

    if rng is None:
        rng = np.random.default_rng()

    C_t = float(rng.uniform(C_min, C_max))
    norm = np.linalg.norm(g)
    if norm <= C_t:
        return g.copy(), C_t
    return g * (C_t / max(norm, 1e-10)), C_t


def analyze_estimator_under_randomization(
    C_min: float,
    C_max: float,
    n_rounds: int = 100,
    n_trials: int = 50,
    seed: int = 42,
) -> dict:
    """Measure how well PassiveParameterEstimator performs under randomized clipping.

    Simulates n_trials independent runs of the estimator.  In each run,
    n_rounds observed norms are generated as C_t ~ U(C_min, C_max) (i.e.
    the norm of a gradient clipped to a fresh random threshold each round).
    The estimator's final estimate of C is compared to the true mean
    C_true_mean = (C_min + C_max) / 2.

    Args:
        C_min:    minimum clipping threshold
        C_max:    maximum clipping threshold
        n_rounds: number of observations per trial (i.e. FL rounds)
        n_trials: number of independent estimation trials
        seed:     RNG seed for reproducibility

    Returns:
        {
          "mean_error": float,    # mean |Ĉ - C_true_mean| over trials
          "std_error":  float,    # std of |Ĉ - C_true_mean| over trials
          "C_true_mean": float,   # (C_min + C_max) / 2
        }
    """
    rng = np.random.default_rng(seed)
    C_true_mean = (C_min + C_max) / 2.0
    errors = []

    for _ in range(n_trials):
        est = PassiveParameterEstimator(history_window=n_rounds)
        for _ in range(n_rounds):
            C_t = float(rng.uniform(C_min, C_max))
            # Observed norm ≈ C_t (gradient exactly at the clipping boundary)
            est.update(C_t)
        C_hat = est.estimate_C()
        errors.append(abs(C_hat - C_true_mean))

    return {
        "mean_error":  float(np.mean(errors)),
        "std_error":   float(np.std(errors)),
        "C_true_mean": C_true_mean,
    }


if __name__ == "__main__":
    result = analyze_estimator_under_randomization(
        C_min=0.5, C_max=2.0, n_rounds=100, n_trials=50, seed=42
    )
    print("Estimator under randomized clipping (C_min=0.5, C_max=2.0):")
    print(f"  True C mean   = {result['C_true_mean']:.3f}")
    print(f"  Estimate error = {result['mean_error']:.4f} ± {result['std_error']:.4f}")
    print(f"  Irreducible std(U) = {(2.0 - 0.5) / (2 * 3 ** 0.5):.4f}")
