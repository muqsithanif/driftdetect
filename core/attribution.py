"""Feature Drift Attribution: Permutation Importance, Consensus Culprit Score, and RAFE."""
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from core.adversarial import AdversarialValidator


@dataclass
class CulpritFeature:
    feature_name: str
    permutation_importance: float  # AUC drop when permuted
    univariate_auc: float          # AUC of single feature alone
    consensus_score: float         # Aggregated rank score [0, 1]
    is_primary_culprit: bool


class DriftAttributionEngine:
    """Isolates the specific features driving train vs test distribution divergence."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def compute_culprits(
        self,
        x_train: np.ndarray,
        x_test: np.ndarray,
        feature_names: List[str],
        base_auc: float,
        n_repeats: int = 3,
    ) -> List[CulpritFeature]:
        """Compute permutation importance and consensus culprit scores."""
        n, m = len(x_train), len(x_test)
        X = np.vstack([x_train, x_test])
        y = np.concatenate([np.zeros(n), np.ones(m)])

        # Train reference discriminator
        clf = RandomForestClassifier(n_estimators=60, max_depth=5, random_state=self.random_state, n_jobs=-1)
        clf.fit(X, y)
        baseline_preds = clf.predict_proba(X)[:, 1]
        baseline_eval_auc = float(roc_auc_score(y, baseline_preds))

        culprits: List[CulpritFeature] = []
        perm_scores = []
        univ_scores = []

        rng = np.random.RandomState(self.random_state)

        for k, name in enumerate(feature_names):
            # 1. Permutation importance (AUC drop)
            auc_drops = []
            for _ in range(n_repeats):
                X_perm = X.copy()
                X_perm[:, k] = rng.permutation(X_perm[:, k])
                preds_perm = clf.predict_proba(X_perm)[:, 1]
                auc_perm = float(roc_auc_score(y, preds_perm))
                auc_drops.append(baseline_eval_auc - auc_perm)
            perm_imp = max(0.0, float(np.mean(auc_drops)))
            perm_scores.append(perm_imp)

            # 2. Univariate AUC
            vals = X[:, k]
            try:
                u_auc = float(roc_auc_score(y, vals))
                if u_auc < 0.5:
                    u_auc = 1.0 - u_auc
            except Exception:
                u_auc = 0.5
            univ_scores.append(u_auc)

        # 3. Consensus Culprit Score C_k (Rank Aggregation)
        perm_ranks = np.argsort(np.argsort(perm_scores)) / max(1, len(feature_names) - 1)
        univ_ranks = np.argsort(np.argsort(univ_scores)) / max(1, len(feature_names) - 1)
        consensus = (perm_ranks + univ_ranks) / 2.0

        for k, name in enumerate(feature_names):
            c_val = float(consensus[k])
            is_culprit = bool(c_val >= 0.70 and perm_scores[k] > 0.005)
            culprits.append(
                CulpritFeature(
                    feature_name=name,
                    permutation_importance=perm_scores[k],
                    univariate_auc=univ_scores[k],
                    consensus_score=c_val,
                    is_primary_culprit=is_culprit,
                )
            )

        # Sort descending by consensus score
        culprits.sort(key=lambda c: c.consensus_score, reverse=True)
        return culprits

    def recursive_feature_elimination(
        self,
        x_train: np.ndarray,
        x_test: np.ndarray,
        feature_names: List[str],
        max_drops: int = 4,
    ) -> List[Tuple[str, float]]:
        """RAFE: Iteratively drop top culprit features and record adversarial AUC decay."""
        validator = AdversarialValidator(n_splits=3, random_state=self.random_state)
        curr_feat_names = list(feature_names)
        curr_x_tr = x_train.copy()
        curr_x_te = x_test.copy()

        initial_res = validator.validate(curr_x_tr, curr_x_te, curr_feat_names)
        trajectory = [("Baseline (All Features)", initial_res.roc_auc)]

        for _ in range(min(max_drops, len(curr_feat_names) - 1)):
            if initial_res.roc_auc <= 0.55:
                break

            culprits = self.compute_culprits(curr_x_tr, curr_x_te, curr_feat_names, initial_res.roc_auc)
            top_drop = culprits[0].feature_name
            drop_idx = curr_feat_names.index(top_drop)

            # Remove feature
            curr_feat_names.pop(drop_idx)
            curr_x_tr = np.delete(curr_x_tr, drop_idx, axis=1)
            curr_x_te = np.delete(curr_x_te, drop_idx, axis=1)

            step_res = validator.validate(curr_x_tr, curr_x_te, curr_feat_names)
            trajectory.append((f"Drop {top_drop}", step_res.roc_auc))
            initial_res = step_res

        return trajectory
