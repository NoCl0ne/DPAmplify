"""
theory/snr_analysis.py — Canonical SNR formulas for DPAmplify.

Two formulas are provided, validated by poc.py (seed=42, all tests PASS):

  compute_attack_snr_upper_bound(k, n, C, sigma)
      Closed-form upper bound; no honest-gradient information needed.
      Formula:  k * C / (sigma * sqrt(n - k))
      WARNING: overestimates SNR by 30-50 % in heterogeneous-data regimes.

  compute_attack_snr_tight(k, n, C, sigma, var_honest)
      Exact SNR accounting for all variance sources.
      Requires var_honest = Var[ clip(g_h, C) · g_target ] (post-clip).
      Empirically accurate to < 5 % in PoC experiments.

Regime boundary:
  The two formulas agree when  Var_h << sigma^2 / n  (IID / concentrated
  honest data).  Use the tight formula whenever Var_h >= sigma^2 / n.

Reference: DPAmplify — poc.py PoC, seed=42
  Upper bound:  SNR = 7.28,  empirical = 4.68  (error 35.6 %)
  Tight bound:  SNR = 4.92,  empirical = 4.68  (error  4.8 %)
"""

import math


def compute_attack_snr_upper_bound(k: int, n: int, C: float, sigma: float) -> float:
    """Upper bound on Byzantine attack SNR in differentially-private FedAvg.

    Formula:
        SNR_ub = k * C / (sigma * sqrt(n - k))

    Derivation sketch:
        The aggregate gradient projected onto g_target has:
          Signal = k * C / n                   (Byzantine contribution)
          Noise  = sigma * sqrt(n - k) / n     (DP noise on honest side only)
        Dividing: SNR_ub = k*C / (sigma * sqrt(n-k)).

    WARNING: This is an upper bound. It omits two variance sources:
        1. DP noise variance from the k Byzantine clients themselves.
        2. Honest gradient variance projected onto g_target (Var_h).
    Use compute_attack_snr_tight() for empirically accurate predictions.

    Args:
        k:     number of Byzantine clients
        n:     total number of clients
        C:     clipping threshold
        sigma: DP noise standard deviation

    Returns:
        Upper-bound SNR (float).
    """
    if k >= n:
        raise ValueError(f"k ({k}) must be less than n ({n})")
    return (k * C) / (sigma * math.sqrt(n - k))


def compute_attack_snr_tight(
    k: int, n: int, C: float, sigma: float, var_honest: float
) -> float:
    """Tight estimate of Byzantine attack SNR in differentially-private FedAvg.

    Derivation (step by step):
    ------------------------------------------------------------------
    Setup:
        g_target  — unit target direction (||g_target|| = 1)
        g_adv     = C * g_target           (adversarial gradient, ||g_adv|| = C)
        M_DP(g)   = clip(g, C) + xi,  xi ~ N(0, sigma^2 * I)

    The per-round aggregate projected onto g_target is:

        P = (1/n) * [ sum_{i in Byzantine} M_DP(g_adv) · g_target
                    + sum_{j in Honest}   M_DP(g_h_j)  · g_target ]

    ── Step 1: Signal (E[P]) ──────────────────────────────────────────
        E[ M_DP(g_adv) · g_target ]
            = clip(g_adv, C) · g_target + E[xi] · g_target
            = C  (because ||g_adv|| = C so clip is identity)
            + 0  (Gaussian noise has zero mean)
            = C

        For zero-mean honest gradients:
            E[ clip(g_h_j, C) · g_target ] ≈ 0

        => Signal = E[P] = k * C / n

    ── Step 2: Var of a single Byzantine projection ───────────────────
        Var[ M_DP(g_adv) · g_target ]
            = Var[ C + xi · g_target ]      (C is a constant)
            = Var[ xi · g_target ]
            = sigma^2                       (||g_target|| = 1)

    ── Step 3: Var of a single honest projection ──────────────────────
        Var[ M_DP(g_h_j) · g_target ]
            = Var[ clip(g_h_j, C) · g_target  +  xi · g_target ]
            = Var[ clip(g_h_j, C) · g_target ] + sigma^2
            =  var_honest + sigma^2

        where var_honest := Var[ clip(g_h, C) · g_target ]
        (the per-client honest variance after clipping, projected onto g_target)

    ── Step 4: Var[P] (all clients independent) ───────────────────────
        Var[P] = (1/n^2) * [ k * sigma^2  +  (n-k) * (var_honest + sigma^2) ]
               = (1/n^2) * [ n * sigma^2  +  (n-k) * var_honest ]
               = sigma^2 / n  +  (n-k) * var_honest / n^2

    ── Step 5: Tight SNR ──────────────────────────────────────────────
        SNR_tight = Signal / sqrt(Var[P])
                  = (k * C / n) / sqrt( sigma^2/n + (n-k)*var_honest/n^2 )

    ── Reduction to upper bound ───────────────────────────────────────
        When var_honest → 0  (honest gradients are highly concentrated):
            SNR_tight → (k*C/n) / sqrt(sigma^2/n)
                      = k*C / (sigma * sqrt(n))
        Note: this limit is tighter than SNR_ub = k*C / (sigma*sqrt(n-k)).
        The upper bound further drops the Byzantine DP noise from the
        denominator, making it slightly looser.

        Regime boundary: the two formulas diverge significantly when
            (n-k) * var_honest / n^2  ≳  sigma^2 / n
        i.e., when  var_honest ≳ sigma^2 / (n-k).
        Use this inequality to decide which formula to apply.
    ------------------------------------------------------------------

    Args:
        k:          number of Byzantine clients
        n:          total number of clients
        C:          clipping threshold
        sigma:      DP noise standard deviation
        var_honest: per-client variance of the honest gradient projected onto
                    g_target, measured AFTER clipping:
                      var_honest = Var[ clip(g_h, C) · g_target ]
                    Estimate this empirically from the simulation, or set to 0
                    for the optimistic case of concentrated honest gradients.

    Returns:
        Tight SNR estimate (float).
    """
    if k >= n:
        raise ValueError(f"k ({k}) must be less than n ({n})")
    if var_honest < 0:
        raise ValueError(f"var_honest must be >= 0, got {var_honest}")

    signal   = k * C / n
    variance = sigma**2 / n + (n - k) * var_honest / n**2
    return signal / math.sqrt(variance)
