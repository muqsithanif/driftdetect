"""End-to-end covariate shift audit, adversarial validation, and alignment pipeline."""
import time
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from core.stats import UnivariateDriftAnalyzer
from core.adversarial import AdversarialValidator
from core.attribution import DriftAttributionEngine
from core.alignment import ImportanceWeightCalculator
from core.visualizer import DriftVisualizer


def generate_synthetic_kaggle_dataset(seed: int = 42):
    """Generate realistic competitive tabular dataset with controlled covariate shifts and ID leakage."""
    rng = np.random.RandomState(seed)
    n_train = 3000
    n_test = 1500

    feature_names = [
        "user_age",
        "monthly_income",
        "debt_to_income_ratio",
        "credit_lines_count",
        "delinquency_history",
        "inquiry_count_6m",
        "revolving_utilization",
        "loan_amount",
        "interest_rate",
        "transaction_id",  # Monotonic ID leakage
    ]

    # Reference Train Distribution (P)
    x_tr = np.column_stack([
        rng.normal(38.0, 10.0, n_train),              # user_age: Normal
        rng.lognormal(8.5, 0.6, n_train),              # monthly_income: Log-normal
        rng.beta(2, 5, n_train) * 1.5,                 # debt_to_income_ratio: Beta
        rng.poisson(6, n_train),                       # credit_lines_count
        rng.binomial(1, 0.12, n_train),                # delinquency_history
        rng.poisson(2, n_train),                       # inquiry_count_6m
        rng.uniform(0.05, 0.85, n_train),              # revolving_utilization
        rng.normal(15000, 4500, n_train),              # loan_amount
        rng.normal(12.5, 2.5, n_train),                # interest_rate
        np.arange(1, n_train + 1),                     # transaction_id (monotonic ID)
    ])

    # Target Test Distribution (Q) - with real-world covariate shifts
    x_te = np.column_stack([
        rng.normal(44.0, 11.5, n_test),               # user_age: SHIFTED (+6 years, higher variance)
        rng.lognormal(8.75, 0.65, n_test),             # monthly_income: SHIFTED (inflation shift)
        rng.beta(3, 4, n_test) * 1.8,                  # debt_to_income_ratio: SHIFTED
        rng.poisson(6, n_test),                        # credit_lines_count: STABLE
        rng.binomial(1, 0.13, n_test),                 # delinquency_history: STABLE
        rng.poisson(2, n_test),                        # inquiry_count_6m: STABLE
        rng.uniform(0.05, 0.85, n_test),               # revolving_utilization: STABLE
        rng.normal(15200, 4600, n_test),               # loan_amount: STABLE
        rng.normal(12.6, 2.5, n_test),                 # interest_rate: STABLE
        np.arange(n_train + 1, n_train + n_test + 1),  # transaction_id: Contrived separator
    ])

    return x_tr, x_te, feature_names


def main():
    print("driftdetect: Running covariate shift and adversarial audit...")

    # 1. Dataset Generation
    print("\n[1/5] Synthesizing Production / Kaggle Tabular Dataset...")
    x_train, x_test, feature_names = generate_synthetic_kaggle_dataset(seed=42)
    print(f"      Reference Train Size     : {x_train.shape[0]} rows x {x_train.shape[1]} features")
    print(f"      Target Test Size        : {x_test.shape[0]} rows x {x_test.shape[1]} features")

    # 2. Univariate Statistical Testing (KS, Wasserstein-1, Sample-Size Calibrated PSI)
    print("\n[2/5] Running Univariate Statistical Drift Analysis (BH-FDR q=0.05)...")
    t0_stat = time.perf_counter()
    univariate_results = UnivariateDriftAnalyzer.analyze_dataset(
        df_reference=x_train,
        df_target=x_test,
        feature_names=feature_names,
        q_fdr=0.05,
    )
    stat_time = (time.perf_counter() - t0_stat) * 1000.0

    print(f"      Statistical Tests Completed in {stat_time:.2f} ms:")
    print("      " + "-" * 75)
    print(f"      {'Feature':<24} | {'KS Stat':<8} | {'p-val':<8} | {'W1 Norm':<8} | {'PSI':<7} | {'Severity'}")
    print("      " + "-" * 75)
    for r in univariate_results:
        print(f"      {r.feature_name:<24} | {r.ks_statistic:<8.3f} | {r.ks_pvalue:<8.2e} | {r.wasserstein_norm:<8.3f} | {r.psi_score:<7.3f} | {r.severity}")
    print("      " + "-" * 75)

    # 3. Multivariate Adversarial Validation
    print("\n[3/5] Training Out-of-Fold Adversarial Validation Discriminator...")
    validator = AdversarialValidator(n_splits=5, use_lightgbm=True, random_state=42)
    t0_adv = time.perf_counter()
    adv_res = validator.validate(x_train, x_test, feature_names)
    adv_time = (time.perf_counter() - t0_adv) * 1000.0

    print(f"      Adversarial CV Completed in {adv_time:.2f} ms")
    if adv_res.trivial_separators:
        print(f"      Leakage Guard Excluded   : {adv_res.trivial_separators} (Trivial Separator)")
    print(f"      Out-of-Fold ROC-AUC      : {adv_res.roc_auc:.4f} (Gini = {adv_res.gini_coefficient:.4f})")
    print(f"      95% Bootstrap CI         : [{adv_res.auc_ci_lower:.4f} - {adv_res.auc_ci_upper:.4f}]")
    print(f"      Overall Drift Severity   : {adv_res.drift_severity}")

    # 4. Feature Drift Attribution & RAFE
    print("\n[4/5] Computing Permutation Attribution & RAFE Decay Trajectory...")
    # Exclude trivial separators from attribution
    non_trivial_feats = [f for f in feature_names if f not in adv_res.trivial_separators]
    feat_indices = [feature_names.index(f) for f in non_trivial_feats]
    x_tr_clean = x_train[:, feat_indices]
    x_te_clean = x_test[:, feat_indices]

    attribution_engine = DriftAttributionEngine(random_state=42)
    culprits = attribution_engine.compute_culprits(x_tr_clean, x_te_clean, non_trivial_feats, adv_res.roc_auc)

    print("      Top Identified Culprit Features (Driving Test Divergence):")
    for i, c in enumerate(culprits[:4]):
        tag = "[PRIMARY CULPRIT]" if c.is_primary_culprit else "[STABLE]"
        print(f"      [{i+1}] {c.feature_name:<24} | Perm Imp: {c.permutation_importance:.4f} | Univ AUC: {c.univariate_auc:.3f} | C_k: {c.consensus_score:.2f} {tag}")

    rafe_trajectory = attribution_engine.recursive_feature_elimination(x_tr_clean, x_te_clean, non_trivial_feats, max_drops=3)
    print("\n      Recursive Adversarial Feature Elimination (RAFE) Trajectory:")
    for step_name, step_auc in rafe_trajectory:
        print(f"      -> {step_name:<28} : Adversarial AUC = {step_auc:.4f}")

    # 5. Adaptive Train-Test Alignment & Verification
    print("\n[5/5] Calculating Importance-Weighted Cross-Validation Weights (IWCV)...")
    weights, ess_ratio = ImportanceWeightCalculator.compute_weights(
        train_oof_probs=adv_res.oof_probabilities,
        n_train=len(x_train),
        n_test=len(x_test),
        lambda_power=0.50,
    )
    print(f"      Effective Sample Size (ESS) : {ess_ratio * 100.0:.1f} % of training set")
    print(f"      Mean Sample Weight         : {np.mean(weights):.2f} (std = {np.std(weights):.2f})")

    # Export Visual Artifacts
    samples_dir = PROJECT_ROOT / "samples"
    samples_dir.mkdir(exist_ok=True)

    dash_path = samples_dir / "drift_dashboard.png"
    top_culprit_name = culprits[0].feature_name
    top_col_idx = feature_names.index(top_culprit_name)

    DriftVisualizer.plot_dashboard(
        univariate_results=univariate_results,
        adversarial_result=adv_res,
        top_culprits=culprits,
        x_train_top=x_train[:, top_col_idx],
        x_test_top=x_test[:, top_col_idx],
        top_feature_name=top_culprit_name,
        save_path=str(dash_path),
    )
    print(f"\n      Saved Drift Dashboard       -> {dash_path}")

    rafe_path = samples_dir / "rafe_decay.png"
    DriftVisualizer.plot_rafe_trajectory(rafe_trajectory, str(rafe_path))
    print(f"      Saved RAFE Decay Trajectory -> {rafe_path}")
    print(f"Drift audit complete. Artifacts saved to: {samples_dir}")


if __name__ == "__main__":
    main()
