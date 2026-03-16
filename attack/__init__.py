"""
DPAmplify attack — two-phase Byzantine attack exploiting the Gaussian
DP mechanism structure.

Phase 1 (estimation, rounds 1..T_est):
  The Byzantine client behaves honestly, observing gradient norms to
  passively estimate the clipping threshold C and noise scale σ.

Phase 2 (attack, rounds T_est+1..):
  The client submits g_adv = C * (g_target / ||g_target||₂).
  Since ||g_adv||₂ = C exactly, the DP mechanism's clip is a no-op, so
  E[M_DP(g_adv)] = g_adv — the adversarial signal is preserved in
  expectation.

Public API:
  attack.parameter_estimator.PassiveParameterEstimator
  attack.gradient_optimizer.GradientOptimizer
  attack.byzantine_client.DPAmplifyClient
"""
