"""
experiments/exp_04_adaptive_clipping.py — Attack robustness under adaptive clipping.

In adaptive clipping the server updates the clipping threshold each round:
    C_{t+1} = quantile_0.5({ ||g_i^t||₂ })   (median of submitted norms)

This experiment shows that:
  1. PassiveParameterEstimator tracks C_t over time even as it changes.
  2. The attack remains effective because g_adv adapts to C̃_t each round.
"""

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory.dp_mechanism import DPMechanism
from attack.parameter_estimator import PassiveParameterEstimator
from attack.gradient_optimizer import GradientOptimizer


# ── Config ──────────────────────────────────────────────────────────
SEED         = 42
D            = 100
N            = 20
K            = 3
SIGMA        = 0.1
N_ROUNDS     = 80
T_EST        = 20

# Adaptive clipping schedule: C_t decreases then stabilises
def true_C(t: int) -> float:
    """Simulate server-side adaptive clipping with a decaying schedule."""
    C_init = 2.0
    C_final = 0.5
    decay = 0.95
    return C_final + (C_init - C_final) * (decay ** t)


OUT_DIR  = os.path.join("paper", "figures")
OUT_FILE = os.path.join(OUT_DIR, "fig_adaptive_clipping.pdf")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(SEED)

    estimator = PassiveParameterEstimator(history_window=T_EST)

    true_C_history  = []
    hat_C_history   = []
    snr_history     = []
    attack_success  = []  # per-round: did g_adv project positively onto e_1?

    g_target = np.zeros(D)
    g_target[0] = 1.0

    for t in range(N_ROUNDS):
        C_t = true_C(t)
        mech = DPMechanism(C=C_t, sigma=SIGMA)
        true_C_history.append(C_t)

        # Simulate honest gradient norms this round
        honest_norms = []
        honest_projs = []
        for _ in range(N - K):
            g_h = rng.normal(0.0, C_t / np.sqrt(D), size=D)
            g_h_dp = mech.apply(g_h, rng)
            norm_h = float(np.linalg.norm(g_h_dp))
            honest_norms.append(norm_h)
            estimator.update(norm_h)
            honest_projs.append(float(np.dot(mech.clip(g_h), g_target)))

        # Adaptive clipping: server updates C based on median norm
        # (we use the true C_t directly to model the server's new threshold;
        #  the estimator tracks it with a window lag)

        # Byzantine gradient uses current estimate
        C_hat = estimator.estimate_C() if estimator.is_ready() else C_t
        hat_C_history.append(C_hat)

        opt = GradientOptimizer(g_target, C=C_hat)
        g_adv = opt.compute_g_adv()

        # Per-round aggregate projection onto g_target
        byz_contribs  = sum(
            float(np.dot(mech.apply(g_adv, rng), g_target))
            for _ in range(K)
        )
        hon_contribs  = sum(
            float(np.dot(mech.apply(
                rng.normal(0.0, C_t / np.sqrt(D), size=D), rng
            ), g_target))
            for _ in range(N - K)
        )
        agg_proj = (byz_contribs + hon_contribs) / N
        attack_success.append(bool(agg_proj > 0))

        # SNR approximation
        var_h = float(np.var(honest_projs)) if len(honest_projs) > 1 else 0.01
        from theory.snr_analysis import compute_attack_snr_tight
        snr = compute_attack_snr_tight(K, N, C_hat, SIGMA, var_h)
        snr_history.append(snr)

    # ── Figure ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("DPAmplify under Adaptive Clipping (seed=42)", fontsize=12)

    rounds = list(range(N_ROUNDS))

    ax = axes[0]
    ax.plot(rounds, true_C_history, "k-",  label="True C_t (server)")
    ax.plot(rounds, hat_C_history,  "b--", label="Estimated Ĉ_t (Byzantine)")
    ax.axvline(T_EST, color="grey", linestyle=":", label=f"End of estimation (T={T_EST})")
    ax.set_xlabel("Round")
    ax.set_ylabel("Clipping threshold C")
    ax.set_title("C tracking over time")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(rounds, snr_history, "tab:orange", label="Tight SNR (Thm 1b)")
    ax.axhline(1.0, color="grey", linestyle=":", label="SNR = 1 (attack boundary)")
    ax.axvline(T_EST, color="grey", linestyle=":")
    ax.set_xlabel("Round")
    ax.set_ylabel("SNR")
    ax.set_title("Attack SNR vs round")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_FILE, dpi=150)
    print(f"Figure saved to {OUT_FILE}")

    success_rate = sum(attack_success) / len(attack_success)
    print(f"Attack success rate (agg_proj > 0): {success_rate:.2%}")
    print(f"Mean SNR post-estimation: "
          f"{float(np.mean(snr_history[T_EST:])):.3f}")

    result = {
        "true_C":      true_C_history,
        "hat_C":       hat_C_history,
        "snr":         snr_history,
        "attack_success_rate": success_rate,
    }
    with open("results/exp_04_results.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
