"""Run the drift audit on one of two datasets.

    python scripts/run_audit.py --dataset injected    # synthetic, drift injected into known columns
    python scripts/run_audit.py --dataset uci-water   # real plant data, two periods compared

Writes results/<dataset>_summary.json and two figures to samples/.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from core.stats import UnivariateDriftAnalyzer
from core.adversarial import AdversarialValidator
from core.attribution import DriftAttributionEngine
from core.alignment import ImportanceWeightCalculator
from core.visualizer import DriftVisualizer

# Column order of water-treatment.data, from water-treatment.names.
ATTRIBUTES = [
    "Q-E", "ZN-E", "PH-E", "DBO-E", "DQO-E", "SS-E", "SSV-E", "SED-E", "COND-E",
    "PH-P", "DBO-P", "SS-P", "SSV-P", "SED-P", "COND-P",
    "PH-D", "DBO-D", "DQO-D", "SS-D", "SSV-D", "SED-D", "COND-D",
    "PH-S", "DBO-S", "DQO-S", "SS-S", "SSV-S", "SED-S", "COND-S",
    "RD-DBO-P", "RD-SS-P", "RD-SED-P", "RD-DBO-S", "RD-DQO-S",
    "RD-DBO-G", "RD-DQO-G", "RD-SS-G", "RD-SED-G",
]


def injected_shift_dataset(seed: int = 42):
    """A credit-style table where the ground truth is known.

    Three columns are shifted on purpose, six are drawn from the same
    distribution on both sides, and a row ID separates the two sides
    trivially. The audit should flag exactly the three shifted columns and
    set the ID aside as leakage.
    """
    rng = np.random.RandomState(seed)
    n_ref, n_tgt = 3000, 1500
    names = [
        "user_age", "monthly_income", "debt_to_income_ratio",   # shifted
        "credit_lines_count", "delinquency_history", "inquiry_count_6m",
        "revolving_utilization", "loan_amount", "interest_rate",
        "transaction_id",                                       # row ID
    ]
    ref = np.column_stack([
        rng.normal(38.0, 10.0, n_ref),
        rng.lognormal(8.5, 0.6, n_ref),
        rng.beta(2, 5, n_ref) * 1.5,
        rng.poisson(6, n_ref),
        rng.binomial(1, 0.12, n_ref),
        rng.poisson(2, n_ref),
        rng.uniform(0.05, 0.85, n_ref),
        rng.normal(15000, 4500, n_ref),
        rng.normal(12.5, 2.5, n_ref),
        np.arange(1, n_ref + 1),
    ])
    tgt = np.column_stack([
        rng.normal(44.0, 11.5, n_tgt),       # older, more spread
        rng.lognormal(8.75, 0.65, n_tgt),    # higher income
        rng.beta(3, 4, n_tgt) * 1.8,         # more leveraged
        rng.poisson(6, n_tgt),
        rng.binomial(1, 0.13, n_tgt),
        rng.poisson(2, n_tgt),
        rng.uniform(0.05, 0.85, n_tgt),
        rng.normal(15200, 4600, n_tgt),
        rng.normal(12.6, 2.5, n_tgt),
        np.arange(n_ref + 1, n_ref + n_tgt + 1),
    ])
    return ref, tgt, names, "reference: 3,000 rows; target: 1,500 rows, drift injected"


def uci_water_periods():
    """Two periods of the UCI Water Treatment Plant data.

    The file stores monthly blocks out of calendar order, so records are
    sorted by date first. The RD-* columns are removal efficiencies derived
    from the other columns and are left out. The periods are records 60-80%
    (January to May 1991) and the last 20% (May to October 1991) in time order.
    """
    data_dir = PROJECT_ROOT / "data" / "uci_water_treatment"
    df = pd.read_csv(data_dir / "water-treatment.data", header=None, na_values=["?"])
    df.columns = ["Date"] + ATTRIBUTES
    df["Date"] = pd.to_datetime(df["Date"].str.replace("D-", "", regex=False), format="%d/%m/%y")
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    features = [c for c in ATTRIBUTES if not c.startswith("RD-")]

    n = len(df)
    ref = df.iloc[int(n * 0.6):int(n * 0.8)]
    tgt = df.iloc[int(n * 0.8):]
    label = (f"reference: {ref['Date'].min():%d %b %Y} to {ref['Date'].max():%d %b %Y} ({len(ref)} records); "
             f"target: {tgt['Date'].min():%d %b %Y} to {tgt['Date'].max():%d %b %Y} ({len(tgt)} records)")
    return ref[features].to_numpy(float), tgt[features].to_numpy(float), features, label


DATASETS = {"injected": injected_shift_dataset, "uci-water": uci_water_periods}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="injected")
    args = parser.parse_args()

    x_ref, x_tgt, names, label = DATASETS[args.dataset]()
    print(f"{args.dataset}: {label}; {len(names)} features")

    # 1. Univariate tests, gated on significance after Benjamini-Hochberg
    univariate = UnivariateDriftAnalyzer.analyze_dataset(x_ref, x_tgt, names, q_fdr=0.05)
    print(f"\n{'feature':<24} {'KS':>6} {'p (comb.)':>10} {'W1/IQR':>7} {'PSI':>6} {'PSI crit':>8}  severity")
    for r in sorted(univariate, key=lambda r: r.p_value):
        print(f"{r.feature_name:<24} {r.ks_statistic:6.3f} {r.p_value:10.2e} {r.wasserstein_norm:7.3f} "
              f"{r.psi_score:6.3f} {r.psi_critical:8.3f}  {r.severity}")

    # 2. Adversarial validation on all columns, with the leakage check first
    adv = AdversarialValidator(n_splits=5, use_lightgbm=True, random_state=42).validate(x_ref, x_tgt, names)
    print(f"\nadversarial AUC {adv.roc_auc:.3f} (95% CI {adv.auc_ci_lower:.3f} to {adv.auc_ci_upper:.3f}), "
          f"{adv.drift_severity}; set aside as trivial separators: {adv.trivial_separators or 'none'}")

    # 3. Which features drive it, and how far removing them goes
    kept = [k for k, name in enumerate(names) if name not in adv.trivial_separators]
    kept_names = [names[k] for k in kept]
    engine = DriftAttributionEngine(random_state=42)
    culprits = engine.compute_culprits(x_ref[:, kept], x_tgt[:, kept], kept_names, adv.roc_auc)
    rafe = engine.recursive_feature_elimination(x_ref[:, kept], x_tgt[:, kept], kept_names, max_drops=3)
    print("\ntop features by consensus rank:")
    for c in culprits[:5]:
        print(f"  {c.feature_name:<24} permutation AUC drop {c.permutation_importance:.3f}, "
              f"univariate AUC {c.univariate_auc:.3f}")
    print("adversarial AUC as the top feature is removed:")
    for step, auc in rafe:
        print(f"  {step:<28} {auc:.3f}")

    # 4. Importance weights for training on the reference data
    weights, ess = ImportanceWeightCalculator.compute_weights(
        train_oof_probs=adv.oof_probabilities, n_train=len(x_ref), n_test=len(x_tgt), lambda_power=0.50,
    )
    print(f"\nimportance weights: effective sample size {ess:.1%} of the reference rows")

    samples = PROJECT_ROOT / "samples"
    samples.mkdir(exist_ok=True)
    top = culprits[0].feature_name
    DriftVisualizer.plot_dashboard(
        univariate_results=univariate,
        adversarial_result=adv,
        top_culprits=culprits,
        x_train_top=x_ref[:, names.index(top)],
        x_test_top=x_tgt[:, names.index(top)],
        top_feature_name=top,
        save_path=str(samples / f"{args.dataset}_dashboard.png"),
    )
    DriftVisualizer.plot_rafe_trajectory(rafe, str(samples / f"{args.dataset}_rafe.png"))

    summary = {
        "dataset": args.dataset,
        "description": label,
        "univariate": [
            {
                "feature": r.feature_name,
                "ks": round(r.ks_statistic, 4),
                "p_value": float(f"{r.p_value:.3g}"),
                "wasserstein_over_iqr": round(r.wasserstein_norm, 4),
                "psi": round(r.psi_score, 4),
                "psi_critical": round(r.psi_critical, 4),
                "severity": r.severity,
            }
            for r in sorted(univariate, key=lambda r: r.p_value)
        ],
        "adversarial": {
            "auc": round(adv.roc_auc, 4),
            "auc_95ci": [round(adv.auc_ci_lower, 4), round(adv.auc_ci_upper, 4)],
            "severity": adv.drift_severity,
            "trivial_separators": adv.trivial_separators,
        },
        "top_features": [c.feature_name for c in culprits[:5]],
        "auc_after_removing_top_features": [[step, round(auc, 4)] for step, auc in rafe],
        "importance_weight_ess": round(ess, 4),
    }
    out = PROJECT_ROOT / "results" / f"{args.dataset}_summary.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out.relative_to(PROJECT_ROOT)} and samples/{args.dataset}_*.png")


if __name__ == "__main__":
    main()
