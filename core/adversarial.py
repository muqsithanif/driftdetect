"""Adversarial Validation: Multivariate distribution divergence testing via discriminators."""
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier

try:
    import lightgbm as lgb
    HAVE_LIGHTGBM = True
except ImportError:
    HAVE_LIGHTGBM = False


@dataclass
class AdversarialValidationResult:
    roc_auc: float
    auc_ci_lower: float
    auc_ci_upper: float
    gini_coefficient: float
    drift_severity: str  # "NEGLIGIBLE", "MILD", "MODERATE", "SEVERE"
    oof_probabilities: np.ndarray  # Out-of-fold p(test)
    labels: np.ndarray             # 0: train, 1: test
    trivial_separators: List[str]


class AdversarialValidator:
    """Trains a discriminative classifier to distinguish train vs test data."""

    def __init__(
        self,
        n_splits: int = 5,
        use_lightgbm: bool = True,
        random_state: int = 42,
    ):
        self.n_splits = n_splits
        self.use_lightgbm = use_lightgbm and HAVE_LIGHTGBM
        self.random_state = random_state

    def check_trivial_separators(
        self,
        x_train: np.ndarray,
        x_test: np.ndarray,
        feature_names: List[str],
        auc_threshold: float = 0.98,
    ) -> List[str]:
        """Detect leakage columns (monotonic IDs, timestamps) that separate datasets trivially."""
        trivial = []
        n, m = len(x_train), len(x_test)
        y = np.concatenate([np.zeros(n), np.ones(m)])

        for k, name in enumerate(feature_names):
            vals = np.concatenate([x_train[:, k], x_test[:, k]])
            finite = np.isfinite(vals)
            if len(np.unique(y[finite])) < 2:
                continue
            score = float(roc_auc_score(y[finite], vals[finite]))
            if max(score, 1.0 - score) >= auc_threshold:
                trivial.append(name)
        return trivial

    def validate(
        self,
        x_train: np.ndarray,
        x_test: np.ndarray,
        feature_names: List[str],
    ) -> AdversarialValidationResult:
        """Run K-fold out-of-fold adversarial validation."""
        # 1. Leakage Guard pre-pass
        trivial = self.check_trivial_separators(x_train, x_test, feature_names)
        valid_indices = [k for k, name in enumerate(feature_names) if name not in trivial]

        if len(valid_indices) == 0:
            valid_indices = list(range(len(feature_names)))

        x_tr_sub = x_train[:, valid_indices]
        x_te_sub = x_test[:, valid_indices]

        n, m = len(x_tr_sub), len(x_te_sub)
        X = np.vstack([x_tr_sub, x_te_sub])
        y = np.concatenate([np.zeros(n), np.ones(m)])

        oof_preds = np.zeros(len(y), dtype=np.float32)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in skf.split(X, y):
            x_fold_tr, y_fold_tr = X[train_idx], y[train_idx]
            x_fold_val, y_fold_val = X[val_idx], y[val_idx]

            if self.use_lightgbm:
                clf = lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    num_leaves=31,
                    scale_pos_weight=float(n / m) if m > 0 else 1.0,
                    random_state=self.random_state,
                    verbose=-1,
                )
            else:
                clf = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=self.random_state,
                    n_jobs=-1,
                )

            clf.fit(x_fold_tr, y_fold_tr)
            probs = clf.predict_proba(x_fold_val)[:, 1]
            oof_preds[val_idx] = probs

        # 2. Metric & Bootstrap Confidence Interval
        overall_auc = float(roc_auc_score(y, oof_preds))
        gini = 2.0 * overall_auc - 1.0

        # Fast bootstrap 95% CI
        rng = np.random.RandomState(self.random_state)
        n_bootstraps = 200
        boot_aucs = []
        for _ in range(n_bootstraps):
            boot_idx = rng.randint(0, len(y), len(y))
            if len(np.unique(y[boot_idx])) > 1:
                boot_aucs.append(float(roc_auc_score(y[boot_idx], oof_preds[boot_idx])))

        if len(boot_aucs) > 0:
            ci_lower = float(np.percentile(boot_aucs, 2.5))
            ci_upper = float(np.percentile(boot_aucs, 97.5))
        else:
            ci_lower, ci_upper = overall_auc, overall_auc

        # Severity classification
        if overall_auc <= 0.55:
            sev = "NEGLIGIBLE"
        elif overall_auc <= 0.65:
            sev = "MILD"
        elif overall_auc <= 0.80:
            sev = "MODERATE"
        else:
            sev = "SEVERE"

        return AdversarialValidationResult(
            roc_auc=overall_auc,
            auc_ci_lower=ci_lower,
            auc_ci_upper=ci_upper,
            gini_coefficient=gini,
            drift_severity=sev,
            oof_probabilities=oof_preds,
            labels=y,
            trivial_separators=trivial,
        )
