"""Adaptive Train-Test Alignment: Importance-Weighted CV and Adversarial Splitters."""
from typing import Tuple, Generator, List, Optional
import numpy as np
from sklearn.model_selection import BaseCrossValidator


class ImportanceWeightCalculator:
    """Computes stabilized importance weights w(x) = q(x)/p(x) for robust training and validation."""

    @staticmethod
    def compute_weights(
        train_oof_probs: np.ndarray,
        n_train: int,
        n_test: int,
        lambda_power: float = 0.50,
        clip_percentile: float = 99.0,
    ) -> Tuple[np.ndarray, float]:
        """Compute stabilized density-ratio sample weights.

        Returns:
            (weights_array, effective_sample_size_ratio)
        """
        # s(x) is p(test | x)
        s = np.clip(train_oof_probs[:n_train], 1e-4, 1.0 - 1e-4)
        base_ratio = (float(n_train) / float(n_test)) * (s / (1.0 - s))

        # 1. Flattening (Shimodaira power parameter)
        w = base_ratio ** lambda_power

        # 2. Clipping extreme outliers
        cap = float(np.percentile(w, clip_percentile))
        w = np.clip(w, 0.0, cap)

        # 3. Normalization so mean(w) = 1.0
        w = w * (n_train / np.sum(w))

        # 4. Effective Sample Size: ESS = (sum w)^2 / sum(w^2)
        ess = float((np.sum(w) ** 2) / np.sum(w ** 2))
        ess_ratio = ess / n_train

        return w, ess_ratio


class AdversarialHoldoutSplitter:
    """Partitions training data such that validation fold is composed of test-most-like samples."""

    def __init__(self, val_fraction: float = 0.20):
        self.val_fraction = val_fraction

    def split(self, X: np.ndarray, train_oof_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (train_indices, val_indices)."""
        n = len(X)
        val_size = int(np.round(n * self.val_fraction))

        # Highest p(test) samples become validation set
        sorted_indices = np.argsort(train_oof_probs[:n])
        train_idx = sorted_indices[:-val_size]
        val_idx = sorted_indices[-val_size:]
        return train_idx, val_idx


class AdversarialStratifiedKFold(BaseCrossValidator):
    """K-Fold cross-validator stratified across test-likeness probability quantiles."""

    def __init__(self, n_splits: int = 5, n_bins: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.n_bins = n_bins
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def _iter_test_indices(self, X=None, y=None, groups=None) -> Generator[np.ndarray, None, None]:
        # 'y' here is train_oof_probs
        if y is None:
            raise ValueError("train_oof_probs must be passed as y to AdversarialStratifiedKFold")

        n = len(X)
        # Bin probabilities into discrete quantiles
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)
        bin_edges = np.percentile(y, quantiles * 100.0)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        binned = np.digitize(y, bin_edges) - 1

        rng = np.random.RandomState(self.random_state)
        indices_per_fold: List[List[int]] = [[] for _ in range(self.n_splits)]

        for b in range(self.n_bins):
            bin_idx = np.where(binned == b)[0]
            rng.shuffle(bin_idx)
            for i, idx in enumerate(bin_idx):
                indices_per_fold[i % self.n_splits].append(int(idx))

        for fold_test in indices_per_fold:
            yield np.array(fold_test, dtype=int)
