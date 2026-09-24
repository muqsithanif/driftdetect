"""Unit tests for adversarial validation."""
import numpy as np
import pytest
from core.adversarial import AdversarialValidator


def test_adversarial_null_distribution():
    rng = np.random.RandomState(42)
    # Identical standard normal distributions
    x_tr = rng.normal(0, 1, (400, 4))
    x_te = rng.normal(0, 1, (200, 4))
    names = ["f1", "f2", "f3", "f4"]

    validator = AdversarialValidator(n_splits=3, use_lightgbm=False, random_state=42)
    res = validator.validate(x_tr, x_te, names)

    # AUC should be near random guessing (0.50 +- noise)
    assert abs(res.roc_auc - 0.50) < 0.12
    assert res.drift_severity in ["NEGLIGIBLE", "MILD"]


def test_trivial_separator_detection():
    rng = np.random.RandomState(42)
    x_tr = rng.normal(0, 1, (300, 2))
    x_te = rng.normal(0, 1, (150, 2))

    # Add perfectly separating column (train: 0..299, test: 300..449)
    tr_id = np.arange(300)[:, None]
    te_id = np.arange(300, 450)[:, None]
    x_tr_leak = np.hstack([x_tr, tr_id])
    x_te_leak = np.hstack([x_te, te_id])
    names = ["num1", "num2", "leak_id"]

    validator = AdversarialValidator()
    trivial = validator.check_trivial_separators(x_tr_leak, x_te_leak, names)
    assert "leak_id" in trivial
