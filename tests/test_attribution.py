"""Unit tests for drift attribution and culprit identification."""
import numpy as np
import pytest
from core.attribution import DriftAttributionEngine


def test_culprit_ranking_identifies_shifted_feature():
    rng = np.random.RandomState(42)
    n_tr, n_te = 500, 250

    # f0, f1, f2 are stationary; f3 is shifted by 3 sigma
    x_tr = rng.normal(0, 1, (n_tr, 4))
    x_te = rng.normal(0, 1, (n_te, 4))
    x_te[:, 3] += 3.0  # Big shift in feature 3
    names = ["stable_a", "stable_b", "stable_c", "shifted_target"]

    engine = DriftAttributionEngine(random_state=42)
    culprits = engine.compute_culprits(x_tr, x_te, names, base_auc=0.85)

    # Shifted feature must be ranked #1
    assert culprits[0].feature_name == "shifted_target"
    assert culprits[0].is_primary_culprit is True
    assert culprits[0].consensus_score >= 0.70
