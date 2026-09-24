"""Unit tests for univariate drift tests: KS, Wasserstein, and PSI."""
import numpy as np
import pytest
from core.stats import UnivariateDriftAnalyzer


def test_identical_distributions_no_drift():
    rng = np.random.RandomState(42)
    x_p = rng.normal(10.0, 2.0, 1000)
    x_q = rng.normal(10.0, 2.0, 1000)

    res = UnivariateDriftAnalyzer.analyze_feature("test_feat", x_p, x_q)

    assert res.is_drifted is False
    assert res.severity == "NONE"
    assert res.ks_statistic < res.ks_critical
    assert res.psi_score < 0.10


def test_shifted_distribution_detected():
    rng = np.random.RandomState(42)
    x_p = rng.normal(10.0, 2.0, 1000)
    # Significant location shift (+4 sigma)
    x_q = rng.normal(18.0, 2.0, 1000)

    res = UnivariateDriftAnalyzer.analyze_feature("shifted_feat", x_p, x_q)

    assert res.is_drifted is True
    assert res.severity == "SEVERE"
    assert res.ks_statistic > 0.50
    assert res.psi_score > 0.25


def test_normalized_wasserstein_scale_invariance():
    rng = np.random.RandomState(42)
    x_p = rng.normal(0, 1, 500)
    x_q = rng.normal(1, 1, 500)
    w_norm_1 = UnivariateDriftAnalyzer.normalized_wasserstein(x_p, x_q)

    # Scale both by 100x
    w_norm_2 = UnivariateDriftAnalyzer.normalized_wasserstein(x_p * 100.0, x_q * 100.0)

    # Normalized Wasserstein should be invariant to linear scale
    np.testing.assert_allclose(w_norm_1, w_norm_2, rtol=1e-3)
