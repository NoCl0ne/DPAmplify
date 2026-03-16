"""
DPAmplify — Minimal Proof of Concept

Validates the three core claims of DPAmplify using only NumPy.
No federated learning framework required.

Tests:
  1. E[M_DP(g_adv)] converges to g_adv (zero-bias claim)
  2. Empirical SNR matches the tight theoretical bound from snr_analysis.py
  3. ‖g_adv‖ after DP is indistinguishable from honest gradient norms

Usage:
  python poc.py

Expected: all three tests PASS.
"""

import numpy as np
from snr_analysis import compute_attack_snr_upper_bound, compute_attack_snr_tight


# ── Core DP functions ───────────────────────────────────────────────

def clip(g: np.ndarray, C: float) -> np.ndarray:
    """Clip gradient to L2 norm C.

    Args:
        g: gradient vector
        C: clipping threshold (must be > 0)

    Returns:
        g * min(1, C / max(‖g‖₂, ε))
    """
    norm = np.linalg.norm(g)
    if norm <= C:
        return g.copy()
    return g * (C / norm)


def apply_dp(g: np.ndarray, C: float, sigma: float,
             rng: np.random.Generator) -> np.ndarray:
    """Apply Gaussian DP mechanism: clip then add noise.

    Args:
        g: gradient vector
        C: clipping threshold
        sigma: noise standard deviation
        rng: numpy random generator (for reproducibility)

    Returns:
        clip(g, C) + N(0, sigma²I)
    """
    return clip(g, C) + rng.normal(0.0, sigma, size=g.shape)


# ── Tests ────────────────────────────────────────────────────────────

def test_expectation(rng: np.random.Generator,
                     d: int, C: float, sigma: float,
                     n_samples: int = 10_000) -> dict:
    """Test 1: E[M_DP(g_adv)] ≈ g_adv.

    Constructs g_adv = C * e_1 (unit vector scaled to norm C).
    Draws n_samples from M_DP(g_adv) and checks that the empirical
    mean is close to g_adv.
    """
    g_target = np.zeros(d)
    g_target[0] = 1.0

    g_adv = C * g_target   # norm = C exactly

    # Sanity check
    assert abs(np.linalg.norm(g_adv) - C) < 1e-10

    samples = np.stack([apply_dp(g_adv, C, sigma, rng)
                        for _ in range(n_samples)])

    empirical_mean = samples.mean(axis=0)
    l2_error = np.linalg.norm(empirical_mean - g_adv)
    threshold = 0.05

    return {
        "g_adv_0":          g_adv[0],
        "empirical_mean_0": empirical_mean[0],
        "l2_error":         l2_error,
        "threshold":        threshold,
        "pass":             l2_error < threshold,
    }


def test_snr(rng: np.random.Generator,
             d: int, n: int, k: int,
             C: float, sigma: float,
             T: int = 300) -> dict:
    """Test 2: Empirical SNR matches the tight theoretical bound.

    Simulates T rounds of FedAvg aggregation with k Byzantine clients
    (DPAmplify) and (n-k) honest clients.  In each round the projection
    of every honest gradient onto g_target (after clipping) is recorded;
    its variance is used to compute the tight SNR via
    compute_attack_snr_tight() from snr_analysis.py.
    """
    g_target = np.zeros(d)
    g_target[0] = 1.0
    g_adv = C * g_target

    # Small honest signal (realistic: honest gradients are noisy)
    mu_honest_scale = 0.1

    projections = []
    # Collect clip(g_h, C) · g_target to estimate var_honest empirically.
    # We use the post-clip value because var_honest in the tight formula
    # is Var[clip(g_h, C) · g_target], not the pre-clip projection.
    honest_clipped_projs = []

    for _ in range(T):
        # Byzantine contributions
        byz_sum = sum(apply_dp(g_adv, C, sigma, rng) for _ in range(k))

        # Honest contributions
        honest_sum = np.zeros(d)
        for _ in range(n - k):
            mu = rng.normal(0.0, mu_honest_scale, size=d)
            g_h = mu + rng.normal(0.0, C / np.sqrt(d), size=d)
            g_h_clipped = clip(g_h, C)
            honest_clipped_projs.append(float(np.dot(g_h_clipped, g_target)))
            honest_sum += g_h_clipped + rng.normal(0.0, sigma, size=d)

        aggregate = (byz_sum + honest_sum) / n
        projections.append(float(np.dot(aggregate, g_target)))

    # Empirical estimate of per-client honest variance (post-clip projection)
    var_honest = float(np.var(honest_clipped_projs))

    proj = np.array(projections)
    snr_empirical = proj.mean() / proj.std()
    snr_upper     = compute_attack_snr_upper_bound(k, n, C, sigma)
    snr_tight     = compute_attack_snr_tight(k, n, C, sigma, var_honest)
    relative_error = abs(snr_empirical - snr_tight) / snr_tight * 100
    threshold_pct  = 10.0   # tight bound should match within 10%

    return {
        "var_honest":       var_honest,
        "snr_upper_bound":  snr_upper,
        "snr_tight":        snr_tight,
        "snr_empirical":    snr_empirical,
        "relative_error_%": relative_error,
        "threshold_%":      threshold_pct,
        "pass":             relative_error < threshold_pct,
    }


def test_norm_indistinguishability(rng: np.random.Generator,
                                   d: int, C: float, sigma: float,
                                   n_samples: int = 1_000) -> dict:
    """Test 3: ‖g_adv‖ after DP is indistinguishable from honest norms.

    Compares the distribution of post-DP norms for g_adv vs honest
    gradients. DPAmplify is stealthy only if these distributions overlap.
    """
    g_target = np.zeros(d)
    g_target[0] = 1.0
    g_adv = C * g_target

    # Adversarial norms
    adv_norms = np.array([
        np.linalg.norm(apply_dp(g_adv, C, sigma, rng))
        for _ in range(n_samples)
    ])

    # Honest gradient norms (gradients with random direction, norm ~ C)
    honest_norms = np.array([
        np.linalg.norm(
            apply_dp(
                rng.normal(0.0, C / np.sqrt(d), size=d),
                C, sigma, rng
            )
        )
        for _ in range(n_samples)
    ])

    mean_diff  = abs(adv_norms.mean() - honest_norms.mean())
    threshold  = 0.15

    return {
        "honest_mean":  honest_norms.mean(),
        "honest_std":   honest_norms.std(),
        "adv_mean":     adv_norms.mean(),
        "adv_std":      adv_norms.std(),
        "mean_diff":    mean_diff,
        "threshold":    threshold,
        "pass":         mean_diff < threshold,
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    # Fixed seed for full reproducibility
    rng = np.random.default_rng(seed=42)

    # Experiment configuration
    d     = 100   # gradient dimension
    n     = 20    # total clients
    k     = 3     # Byzantine clients
    C     = 1.0   # clipping threshold
    sigma = 0.1   # DP noise standard deviation

    print("DPAmplify — Minimal Proof of Concept")
    print(f"Config: d={d}, n={n}, k={k}, C={C}, sigma={sigma}")
    print()

    # ── Test 1 ────────────────────────────────────────────────────
    r1 = test_expectation(rng, d, C, sigma)
    print("=" * 55)
    print("TEST 1: E[M_DP(g_adv)] converges to g_adv")
    print("=" * 55)
    print(f"  g_adv[0]              = {r1['g_adv_0']:.6f}")
    print(f"  E[M_DP(g_adv)][0]     = {r1['empirical_mean_0']:.6f}")
    print(f"  L2 error              = {r1['l2_error']:.6f}")
    print(f"  Threshold             = {r1['threshold']}")
    print(f"  Result                = {'PASS' if r1['pass'] else 'FAIL'}")

    # ── Test 2 ────────────────────────────────────────────────────
    r2 = test_snr(rng, d, n, k, C, sigma)
    print()
    print("=" * 55)
    print("TEST 2: Empirical SNR vs tight theoretical bound")
    print("=" * 55)
    print(f"  var_honest (estimated)= {r2['var_honest']:.6f}")
    print(f"  SNR upper bound       = {r2['snr_upper_bound']:.4f}")
    print(f"  SNR tight bound       = {r2['snr_tight']:.4f}")
    print(f"  Empirical SNR         = {r2['snr_empirical']:.4f}")
    print(f"  Relative error        = {r2['relative_error_%']:.1f}%")
    print(f"  Threshold             = {r2['threshold_%']}%")
    print(f"  Result                = {'PASS' if r2['pass'] else 'FAIL'}")

    # ── Test 3 ────────────────────────────────────────────────────
    r3 = test_norm_indistinguishability(rng, d, C, sigma)
    print()
    print("=" * 55)
    print("TEST 3: Norm indistinguishability (evasion check)")
    print("=" * 55)
    print(f"  Honest norms mean±std = "
          f"{r3['honest_mean']:.4f} ± {r3['honest_std']:.4f}")
    print(f"  Adversarial  mean±std = "
          f"{r3['adv_mean']:.4f} ± {r3['adv_std']:.4f}")
    print(f"  Mean difference       = {r3['mean_diff']:.4f}")
    print(f"  Threshold             = {r3['threshold']}")
    print(f"  Result                = {'PASS' if r3['pass'] else 'FAIL'}")

    # ── Summary ───────────────────────────────────────────────────
    all_pass = r1["pass"] and r2["pass"] and r3["pass"]
    print()
    print("=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"  Test 1 (zero-bias expectation): "
          f"{'PASS' if r1['pass'] else 'FAIL'}")
    print(f"  Test 2 (SNR tight bound):       "
          f"{'PASS' if r2['pass'] else 'FAIL'}")
    print(f"  Test 3 (norm evasion):          "
          f"{'PASS' if r3['pass'] else 'FAIL'}")
    print()
    if all_pass:
        print("  ALL TESTS PASSED.")
        print("  Core mathematical claim is empirically validated.")
        print("  Safe to proceed with full implementation.")
    else:
        print("  ONE OR MORE TESTS FAILED.")
        print("  Review the mathematics before proceeding.")
        print("  Check which test failed and why.")

    return all_pass


if __name__ == "__main__":
    main()
