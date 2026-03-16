"""
tests/test_dp_mechanism.py — Unit tests for the DPAmplify core modules.

Run with:
    pytest tests/ -v
"""

import numpy as np
import pytest

from theory.dp_mechanism import DPMechanism
from attack.gradient_optimizer import GradientOptimizer
from attack.parameter_estimator import PassiveParameterEstimator


# ── DPMechanism — clip behaviour ─────────────────────────────────────

def test_clip_below_threshold():
    """Gradient with norm < C must be returned unchanged."""
    g = np.array([0.5, 0.0, 0.0])
    result = DPMechanism(C=1.0, sigma=0.1).clip(g)
    assert np.allclose(result, g)


def test_clip_above_threshold():
    """Gradient with norm > C must be scaled to norm exactly C."""
    g = np.array([2.0, 0.0, 0.0])
    result = DPMechanism(C=1.0, sigma=0.1).clip(g)
    assert abs(np.linalg.norm(result) - 1.0) < 1e-6


def test_clip_exactly_at_threshold():
    """Gradient with norm == C must be returned unchanged."""
    g = np.array([1.0, 0.0, 0.0])
    result = DPMechanism(C=1.0, sigma=0.1).clip(g)
    assert np.allclose(result, g)


# ── DPMechanism — noise properties ───────────────────────────────────

def test_noise_zero_mean():
    """Averaged noise across 10 000 samples must be close to zero."""
    rng = np.random.default_rng(0)
    mech = DPMechanism(C=1.0, sigma=0.1)
    g = np.zeros(10)
    samples = mech.sample_outputs(g, 10_000, rng)
    assert np.abs(samples.mean()) < 0.01


def test_expected_output_equals_clip():
    """expected_output must equal clip (E[ξ] = 0)."""
    mech = DPMechanism(C=1.0, sigma=0.1)
    g = np.array([2.0, 1.0, 0.5])
    assert np.allclose(mech.expected_output(g), mech.clip(g))


def test_sample_outputs_shape():
    """sample_outputs must return shape (n_samples, d)."""
    mech = DPMechanism(C=1.0, sigma=0.1)
    rng = np.random.default_rng(7)
    samples = mech.sample_outputs(np.zeros(5), 100, rng)
    assert samples.shape == (100, 5)


# ── GradientOptimizer — adversarial gradient ─────────────────────────

def test_g_adv_norm_equals_C():
    """||g_adv||₂ must equal C exactly (so clip is the identity)."""
    opt = GradientOptimizer(np.array([3.0, 4.0]), C=1.0)
    assert abs(np.linalg.norm(opt.compute_g_adv()) - 1.0) < 1e-6


def test_g_adv_not_clipped():
    """clip(g_adv, C) must equal g_adv (no attenuation)."""
    mech = DPMechanism(C=1.0, sigma=0.1)
    opt = GradientOptimizer(np.array([3.0, 4.0]), C=1.0)
    g_adv = opt.compute_g_adv()
    assert np.allclose(mech.clip(g_adv), g_adv)


# ── PassiveParameterEstimator ─────────────────────────────────────────

def test_estimator_ready_after_window():
    """is_ready() must be True after history_window observations."""
    est = PassiveParameterEstimator(history_window=5)
    assert not est.is_ready()
    for v in [0.9, 1.0, 1.1, 0.95, 1.05]:
        est.update(v)
    assert est.is_ready()


# ── DPMechanism — validation errors ──────────────────────────────────

def test_invalid_C_raises():
    """C <= 0 must raise ValueError."""
    with pytest.raises(ValueError):
        DPMechanism(C=0.0, sigma=0.1)


def test_invalid_sigma_raises():
    """sigma <= 0 must raise ValueError."""
    with pytest.raises(ValueError):
        DPMechanism(C=1.0, sigma=0.0)
