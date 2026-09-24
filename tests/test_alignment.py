"""Unit tests for importance weighting and adversarial CV splitters."""
import numpy as np
import pytest
from core.alignment import ImportanceWeightCalculator, AdversarialHoldoutSplitter


def test_importance_weights_properties():
    n_train = 400
    n_test = 200
    # Dummy OOF probabilities
    oof_probs = np.linspace(0.1, 0.9, n_train + n_test)

    weights, ess_ratio = ImportanceWeightCalculator.compute_weights(
        train_oof_probs=oof_probs,
        n_train=n_train,
        n_test=n_test,
        lambda_power=0.5,
    )

    # Total sum of weights must equal n_train
    np.testing.assert_allclose(np.sum(weights), n_train, rtol=1e-3)
    # Effective sample size must be positive and bounded by 1.0
    assert 0.0 < ess_ratio <= 1.0


def test_adversarial_holdout_split_fractions():
    n_train = 500
    X = np.random.randn(n_train, 3)
    oof_probs = np.random.uniform(0.1, 0.9, n_train)

    splitter = AdversarialHoldoutSplitter(val_fraction=0.20)
    tr_idx, val_idx = splitter.split(X, oof_probs)

    assert len(tr_idx) == 400
    assert len(val_idx) == 100
    # No overlap
    assert len(set(tr_idx).intersection(set(val_idx))) == 0
