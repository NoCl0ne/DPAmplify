"""
experiments/exp_01_snr_validation.py — Validate both SNR bounds empirically.

Generates Figure 1 (4-panel) saved to paper/figures/fig_snr_validation.pdf.

Config:
    d=100, C=1.0, seed=42
    sigma_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    k_values     = [1, 2, 3, 5, 7, 10]  (n=20 throughout)
    n_samples    = 5000 per (sigma, k) combination

For each (sigma, k):
  1. g_adv = C * e_1
  2. Sample 5000 outputs of M_DP(g_adv), project onto g_target = e_1.
  3. Empirical SNR = mean(projections) / std(projections)
  4. SNR_upper = compute_attack_snr_upper_bound(k, n, C, sigma)
  5. Estimate var_honest from 5000 honest gradient samples (post-clip).
  6. SNR_tight  = compute_attack_snr_tight(k, n, C, sigma, var_honest)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory.dp_mechanism import DPMechanism
from theory.snr_analysis import (
    compute_attack_snr_upper_bound,
    compute_attack_snr_tight,
)

# ── Config ─────────────────────────────────────────────────────────────
SEED         = 42
D            = 100
N            = 20
C            = 1.0
N_SAMPLES    = 5_000
MU_HONEST_SCALE = 0.1

SIGMA_VALUES = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
K_VALUES     = [1, 2, 3, 5, 7, 10]

OUT_DIR = os.path.join("paper", "figures")
OUT_FILE = os.path.join(OUT_DIR, "fig_snr_validation.pdf")


def _unit_vector(d: int, idx: int = 0) -> np.ndarray:
    v = np.zeros(d)
    v[idx] = 1.0
    return v


def _estimate_var_honest(
    mech: DPMechanism, g_target: np.ndarray, n_samples: int,
    rng: np.random.Generator, mu_scale: float
) -> float:
    """Estimate Var[clip(g_h, C) · g_target] from random honest gradients."""
    d = len(g_target)
    projs = []
    for _ in range(n_samples):
        mu = rng.normal(0.0, mu_scale, size=d)
        g_h = mu + rng.normal(0.0, C / np.sqrt(d), size=d)
        g_h_clipped = mech.clip(g_h)
        projs.append(float(np.dot(g_h_clipped, g_target)))
    return float(np.var(projs))


def compute_metrics(sigma: float, k: int, rng: np.random.Generator) -> dict:
    """Compute empirical and theoretical SNR for one (sigma, k) pair."""
    mech = DPMechanism(C=C, sigma=sigma)
    g_target = _unit_vector(D)
    g_adv    = C * g_target

    # Empirical SNR: project M_DP(g_adv) samples onto g_target
    samples = mech.sample_outputs(g_adv, N_SAMPLES, rng)
    projs   = samples @ g_target          # shape (N_SAMPLES,)
    snr_emp = projs.mean() / (projs.std() + 1e-12)

    # Theoretical bounds
    snr_upper = compute_attack_snr_upper_bound(k, N, C, sigma)
    var_h     = _estimate_var_honest(mech, g_target, N_SAMPLES, rng, MU_HONEST_SCALE)
    snr_tight = compute_attack_snr_tight(k, N, C, sigma, var_h)

    rel_err_upper = abs(snr_emp - snr_upper) / (snr_upper + 1e-12) * 100
    rel_err_tight = abs(snr_emp - snr_tight) / (snr_tight + 1e-12) * 100

    return {
        "snr_emp":        snr_emp,
        "snr_upper":      snr_upper,
        "snr_tight":      snr_tight,
        "rel_err_upper%": rel_err_upper,
        "rel_err_tight%": rel_err_tight,
        "var_honest":     var_h,
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ── Collect metrics ─────────────────────────────────────────────
    # Vary sigma (k=3 fixed)
    K_FIXED    = 3
    SIGMA_FIXED = 0.1

    sigma_metrics = {s: compute_metrics(s, K_FIXED, rng) for s in SIGMA_VALUES}
    k_metrics     = {k: compute_metrics(SIGMA_FIXED, k, rng) for k in K_VALUES}

    # ── Figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("DPAmplify SNR Validation (PoC, seed=42)", fontsize=13)

    # Panel 1: SNR vs sigma (k=3)
    ax = axes[0, 0]
    ax.plot(SIGMA_VALUES, [sigma_metrics[s]["snr_emp"]   for s in SIGMA_VALUES],
            "o-", label="Empirical",  color="black")
    ax.plot(SIGMA_VALUES, [sigma_metrics[s]["snr_upper"] for s in SIGMA_VALUES],
            "s--", label="Upper bound (Thm 1a)", color="tab:red")
    ax.plot(SIGMA_VALUES, [sigma_metrics[s]["snr_tight"] for s in SIGMA_VALUES],
            "^-", label="Tight bound (Thm 1b)", color="tab:blue")
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8, label="SNR = 1")
    ax.set_xlabel("σ (DP noise std)")
    ax.set_ylabel("SNR")
    ax.set_title(f"SNR vs σ  (k={K_FIXED}, n={N})")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: SNR vs k (sigma=0.1)
    ax = axes[0, 1]
    ax.plot(K_VALUES, [k_metrics[k]["snr_emp"]   for k in K_VALUES],
            "o-", label="Empirical",  color="black")
    ax.plot(K_VALUES, [k_metrics[k]["snr_upper"] for k in K_VALUES],
            "s--", label="Upper bound (Thm 1a)", color="tab:red")
    ax.plot(K_VALUES, [k_metrics[k]["snr_tight"] for k in K_VALUES],
            "^-", label="Tight bound (Thm 1b)", color="tab:blue")
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8, label="SNR = 1")
    ax.set_xlabel("k (Byzantine clients)")
    ax.set_ylabel("SNR")
    ax.set_title(f"SNR vs k  (σ={SIGMA_FIXED}, n={N})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Relative error vs sigma
    ax = axes[1, 0]
    ax.plot(SIGMA_VALUES, [sigma_metrics[s]["rel_err_upper%"] for s in SIGMA_VALUES],
            "s--", label="Upper bound error", color="tab:red")
    ax.plot(SIGMA_VALUES, [sigma_metrics[s]["rel_err_tight%"] for s in SIGMA_VALUES],
            "^-",  label="Tight bound error", color="tab:blue")
    ax.axhline(10, color="grey", linestyle=":", linewidth=0.8, label="10 % threshold")
    ax.set_xlabel("σ (DP noise std)")
    ax.set_ylabel("Relative error (%)")
    ax.set_title("Prediction error vs σ")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Distribution of projected samples (sigma=0.1, k=3)
    rng2 = np.random.default_rng(SEED + 1)
    mech_ref = DPMechanism(C=C, sigma=SIGMA_FIXED)
    g_target_ref = _unit_vector(D)
    g_adv_ref    = C * g_target_ref
    samples_ref  = mech_ref.sample_outputs(g_adv_ref, N_SAMPLES, rng2)
    projs_ref    = samples_ref @ g_target_ref

    ax = axes[1, 1]
    ax.hist(projs_ref, bins=60, density=True, color="steelblue", alpha=0.7,
            edgecolor="none", label="M_DP(g_adv) · g_target")
    ax.axvline(projs_ref.mean(), color="black",   linestyle="-",  label=f"Mean={projs_ref.mean():.3f}")
    ax.axvline(C,                color="tab:red", linestyle="--", label=f"C={C}")
    ax.set_xlabel("Projection onto g_target")
    ax.set_ylabel("Density")
    ax.set_title(f"Sample distribution  (σ={SIGMA_FIXED}, k={K_FIXED})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_FILE, dpi=150)
    print(f"Figure saved to {OUT_FILE}")

    # ── Summary table ───────────────────────────────────────────────
    print(f"\nSNR summary (k={K_FIXED}, n={N}):")
    print(f"{'sigma':>8}  {'Empirical':>10}  {'Upper':>8}  {'Tight':>8}  "
          f"{'Err_up%':>8}  {'Err_tgt%':>8}")
    for s in SIGMA_VALUES:
        m = sigma_metrics[s]
        print(f"{s:8.3f}  {m['snr_emp']:10.3f}  {m['snr_upper']:8.3f}  "
              f"{m['snr_tight']:8.3f}  {m['rel_err_upper%']:8.1f}  "
              f"{m['rel_err_tight%']:8.1f}")


if __name__ == "__main__":
    main()
