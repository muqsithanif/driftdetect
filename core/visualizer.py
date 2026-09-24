"""Visualization dashboard for covariate shift diagnostics, ROC curves, and RAFE decay."""
from typing import List, Tuple, Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from core.stats import UnivariateDriftResult
from core.adversarial import AdversarialValidationResult
from core.attribution import CulpritFeature


class DriftVisualizer:
    """Renders comprehensive multi-panel diagnostic reports for distribution drift."""

    @classmethod
    def plot_dashboard(
        cls,
        univariate_results: List[UnivariateDriftResult],
        adversarial_result: AdversarialValidationResult,
        top_culprits: List[CulpritFeature],
        x_train_top: np.ndarray,
        x_test_top: np.ndarray,
        top_feature_name: str,
        save_path: str,
    ) -> None:
        """Plot a 4-panel diagnostic dashboard: ROC, PSI bars, ECDF overlay, and OOF probabilities."""
        fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=120)

        # 1. Adversarial ROC Curve
        ax_roc = axes[0, 0]
        fpr, tpr, _ = roc_curve(adversarial_result.labels, adversarial_result.oof_probabilities)
        ax_roc.plot(fpr, tpr, color="#2b5c8f", lw=2.5,
                    label=f"OOF Discriminator (AUC = {adversarial_result.roc_auc:.3f})")
        ax_roc.plot([0, 1], [0, 1], color="gray", lw=1.2, linestyle="--", label="Random Manifold (AUC = 0.50)")
        ax_roc.fill_between([0, 1], [0, 1], alpha=0.08, color="gray")

        ax_roc.set_title(
            f"Adversarial Validation Divergence\nSeverity: {adversarial_result.drift_severity} "
            f"[95% CI: {adversarial_result.auc_ci_lower:.3f} - {adversarial_result.auc_ci_upper:.3f}]",
            fontsize=10, weight="bold"
        )
        ax_roc.set_xlabel("False Positive Rate (FPR)", fontsize=9)
        ax_roc.set_ylabel("True Positive Rate (TPR)", fontsize=9)
        ax_roc.legend(loc="lower right", fontsize=8)
        ax_roc.grid(alpha=0.3)

        # 2. Population Stability Index (PSI) Bar Chart
        ax_psi = axes[0, 1]
        feat_names = [r.feature_name for r in univariate_results]
        psi_vals = [r.psi_score for r in univariate_results]
        colors = ["#d90429" if r.psi_score > 0.25 else "#f77f00" if r.psi_score > 0.10 else "#2a9d8f" for r in univariate_results]

        y_pos = np.arange(len(feat_names))
        ax_psi.barh(y_pos, psi_vals, color=colors, edgecolor="black", alpha=0.85)
        ax_psi.axvline(0.10, color="#f77f00", linestyle=":", lw=1.5, label="Mild Drift (0.10)")
        ax_psi.axvline(0.25, color="#d90429", linestyle="--", lw=1.5, label="Significant Drift (0.25)")
        ax_psi.set_yticks(y_pos)
        ax_psi.set_yticklabels(feat_names, fontsize=8)
        ax_psi.invert_yaxis()
        ax_psi.set_xlabel("Population Stability Index (PSI)", fontsize=9)
        ax_psi.set_title("Univariate Feature Stability Index", fontsize=10, weight="bold")
        ax_psi.legend(loc="lower right", fontsize=8)
        ax_psi.grid(axis="x", alpha=0.3)

        # 3. Empirical CDF (ECDF) Overlay for Top Culprit
        ax_ecdf = axes[1, 0]
        x_tr_sorted = np.sort(x_train_top)
        x_te_sorted = np.sort(x_test_top)
        y_tr = np.arange(1, len(x_tr_sorted) + 1) / len(x_tr_sorted)
        y_te = np.arange(1, len(x_te_sorted) + 1) / len(x_te_sorted)

        ax_ecdf.plot(x_tr_sorted, y_tr, color="#1d3557", lw=2, label="Train Distribution (P)")
        ax_ecdf.plot(x_te_sorted, y_te, color="#e63946", lw=2, linestyle="-.", label="Test Distribution (Q)")
        ax_ecdf.set_title(f"ECDF Shift: '{top_feature_name}' (Top Culprit)", fontsize=10, weight="bold")
        ax_ecdf.set_xlabel("Feature Value", fontsize=9)
        ax_ecdf.set_ylabel("Cumulative Probability", fontsize=9)
        ax_ecdf.legend(loc="lower right", fontsize=8)
        ax_ecdf.grid(alpha=0.3)

        # 4. Out-of-fold Test Probability Distribution
        ax_oof = axes[1, 1]
        labels = adversarial_result.labels
        oof = adversarial_result.oof_probabilities
        train_oof = oof[labels == 0]
        test_oof = oof[labels == 1]

        ax_oof.hist(train_oof, bins=25, alpha=0.6, color="#457b9d", label="Train Rows (z=0)", density=True, edgecolor="black")
        ax_oof.hist(test_oof, bins=25, alpha=0.6, color="#e63946", label="Test Rows (z=1)", density=True, edgecolor="black")
        ax_oof.set_title("Out-of-Fold p(test | x) Density Separation", fontsize=10, weight="bold")
        ax_oof.set_xlabel("Predicted Probability p(test)", fontsize=9)
        ax_oof.set_ylabel("Density", fontsize=9)
        ax_oof.legend(loc="upper center", fontsize=8)
        ax_oof.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
        plt.close(fig)

    @classmethod
    def plot_rafe_trajectory(cls, trajectory: List[Tuple[str, float]], save_path: str) -> None:
        """Plot Recursive Adversarial Feature Elimination (RAFE) decay curve."""
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
        steps, aucs = zip(*trajectory)
        x_indices = np.arange(len(steps))

        ax.plot(x_indices, aucs, "-o", color="#d90429", lw=2.5, markersize=7)
        ax.axhline(0.55, color="green", linestyle="--", lw=1.5, label="Aligned Threshold (AUC <= 0.55)")

        ax.set_xticks(x_indices)
        ax.set_xticklabels(steps, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Adversarial ROC-AUC", fontsize=9)
        ax.set_title("Recursive Adversarial Feature Elimination (RAFE) Trajectory", fontsize=10, weight="bold")
        ax.set_ylim([0.45, 1.02])
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
