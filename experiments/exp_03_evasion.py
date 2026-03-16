"""
experiments/exp_03_evasion.py — Evasion test against Byzantine-robust aggregators.

Tests whether g_adv evades:
  Krum:          Does g_adv win the Krum selection criterion?
  TrimmedMean:   Do g_adv coordinates fall within the non-trimmed range?

For each round the script generates a fresh pool of (n_clients - n_byzantine)
honest gradients and n_byzantine adversarial gradients, then checks whether
the aggregator accepts the adversarial input.
"""

import argparse
import json
import os

import numpy as np

from theory.dp_mechanism import DPMechanism
from attack.gradient_optimizer import GradientOptimizer
from fl_system.aggregators.krum import krum_aggregate
from fl_system.aggregators.trimmed_mean import trimmed_mean_aggregate


# ── Helpers ───────────────────────────────────────────────────────────

def _honest_gradient(
    d: int, C: float, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Simulate an honest DP gradient: random direction, DP noise applied."""
    g_h = rng.normal(0.0, C / np.sqrt(d), size=d)
    mech = DPMechanism(C=C, sigma=sigma)
    return mech.apply(g_h, rng)


def _adv_gradient(
    d: int, C: float, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Adversarial gradient g_adv = C * e_1, then DP-noised."""
    g_target = np.zeros(d)
    g_target[0] = 1.0
    opt = GradientOptimizer(g_target, C=C)
    g_adv = opt.compute_g_adv()
    mech = DPMechanism(C=C, sigma=sigma)
    return mech.apply(g_adv, rng)


# ── Per-round evasion checks ──────────────────────────────────────────

def krum_evasion_check(
    gradients: list, adv_indices: list, f: int
) -> bool:
    """Return True if any adversarial gradient wins Krum selection."""
    n = len(gradients)
    n_neighbours = n - f - 2
    if n_neighbours < 1:
        return False

    import numpy as np
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diff = gradients[i] - gradients[j]
            d2 = float(np.dot(diff, diff))
            distances[i, j] = d2
            distances[j, i] = d2

    scores = []
    for i in range(n):
        row = distances[i].copy()
        row[i] = np.inf
        scores.append(np.sort(row)[:n_neighbours].sum())

    winner = int(np.argmin(scores))
    return winner in adv_indices


def trimmed_evasion_check(
    gradients: list, adv_indices: list, beta: float
) -> bool:
    """Return True if all adversarial coordinates fall within the non-trimmed range."""
    import math, numpy as np
    n = len(gradients)
    trim = math.floor(beta * n)
    G = np.stack(gradients, axis=0)  # (n, d)
    G_sorted = np.sort(G, axis=0)

    lo = G_sorted[trim]       # lower trim boundary per coord
    hi = G_sorted[n - trim - 1]  # upper trim boundary per coord

    for i in adv_indices:
        g = gradients[i]
        # Gradient evades if all coordinates are within [lo, hi]
        if np.all(g >= lo) and np.all(g <= hi):
            return True
    return False


# ── Main ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DPAmplify evasion experiment")
    p.add_argument("--aggregator",  type=str, required=True,
                   choices=["krum", "trimmed_mean"])
    p.add_argument("--n_clients",   type=int, default=20)
    p.add_argument("--n_byzantine", type=int, default=3)
    p.add_argument("--n_rounds",    type=int, default=50)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    n = args.n_clients
    k = args.n_byzantine
    n_honest = n - k
    D = 100
    C = 1.0
    SIGMA = 0.1
    BETA = 0.1

    adv_indices = list(range(n_honest, n))
    detection_rounds = 0
    evasion_rounds   = 0

    for rnd in range(args.n_rounds):
        honest_grads = [_honest_gradient(D, C, SIGMA, rng) for _ in range(n_honest)]
        adv_grads    = [_adv_gradient(D, C, SIGMA, rng)    for _ in range(k)]
        all_grads    = honest_grads + adv_grads

        if args.aggregator == "krum":
            evades = krum_evasion_check(all_grads, adv_indices, f=k)
        else:
            evades = trimmed_evasion_check(all_grads, adv_indices, beta=BETA)

        if evades:
            evasion_rounds += 1
        else:
            detection_rounds += 1

    evasion_rate   = evasion_rounds   / args.n_rounds
    detection_rate = detection_rounds / args.n_rounds

    # Accuracy degradation: proxy = fraction of rounds where adversary evades
    accuracy_degradation = evasion_rate

    result = {
        "aggregator":          args.aggregator,
        "n_clients":           n,
        "n_byzantine":         k,
        "n_rounds":            args.n_rounds,
        "evasion_rounds":      evasion_rounds,
        "detection_rounds":    detection_rounds,
        "evasion_rate":        evasion_rate,
        "detection_rate":      detection_rate,
        "accuracy_degradation": accuracy_degradation,
    }

    print(f"Aggregator:         {args.aggregator}")
    print(f"Byzantine clients:  {k}/{n}")
    print(f"Evasion rate:       {evasion_rate:.2%}  ({evasion_rounds}/{args.n_rounds} rounds)")
    print(f"Detection rate:     {detection_rate:.2%}  ({detection_rounds}/{args.n_rounds} rounds)")
    print(f"Accuracy degradation proxy: {accuracy_degradation:.2%}")

    os.makedirs("results", exist_ok=True)
    out_path = f"results/exp_03_{args.aggregator}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
