"""Statistical Covariate Shift Quantification: Kolmogorov-Smirnov, Wasserstein-1, and Calibrated PSI."""
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
from scipy import stats


@dataclass
class UnivariateDriftResult:
    feature_name: str
    ks_statistic: float
    ks_pvalue: float
    ks_critical: float
    wasserstein_norm: float
    psi_score: float
    psi_critical: float
    p_value: float     # combined KS/PSI p-value, before the FDR step
    is_drifted: bool
    severity: str  # "NONE", "LOW", "MODERATE", "SEVERE"


class UnivariateDriftAnalyzer:
    """Computes rigorous univariate two-sample statistical tests with sample-size aware calibration."""

    @staticmethod
    def kolmogorov_smirnov(x_p: np.ndarray, x_q: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float]:
        """Compute two-sample Kolmogorov-Smirnov D statistic, asymptotic p-value, and critical value."""
        res = stats.ks_2samp(x_p, x_q)
        n, m = len(x_p), len(x_q)
        c_alpha = np.sqrt(-0.5 * np.log(alpha / 2.0))
        d_crit = c_alpha * np.sqrt((n + m) / (n * m))
        return float(res.statistic), float(res.pvalue), float(d_crit)

    @staticmethod
    def normalized_wasserstein(x_p: np.ndarray, x_q: np.ndarray) -> float:
        """Compute Wasserstein-1 (Earth Mover's Distance) normalized by reference sample IQR."""
        w1 = stats.wasserstein_distance(x_p, x_q)
        iqr = float(stats.iqr(x_p))
        if iqr > 1e-6:
            return float(w1 / iqr)
        std_p = float(np.std(x_p))
        if std_p > 1e-6:
            return float(w1 / std_p)
        return float(w1)

    @staticmethod
    def population_stability_index(
        x_p: np.ndarray,
        x_q: np.ndarray,
        num_bins: int = 10,
        alpha: float = 0.05,
        eps: float = 1e-4,
    ) -> Tuple[float, float, float]:
        """PSI, its critical value at this sample size, and its p-value under no drift."""
        n, m = len(x_p), len(x_q)
        # Construct equal-mass bins from reference percentiles
        quantiles = np.linspace(0.0, 1.0, num_bins + 1)
        bin_edges = np.percentile(x_p, quantiles * 100.0)
        # Deduplicate bin edges if data has repeated values
        bin_edges = np.unique(bin_edges)
        b_actual = len(bin_edges) - 1

        if b_actual < 2:
            return 0.0, 0.10, 1.0

        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        counts_p, _ = np.histogram(x_p, bins=bin_edges)
        counts_q, _ = np.histogram(x_q, bins=bin_edges)

        # Proportions with Laplace smoothing
        prop_p = (counts_p + eps) / (n + b_actual * eps)
        prop_q = (counts_q + eps) / (m + b_actual * eps)

        # PSI = sum (Q - P) * ln(Q / P)
        psi_val = float(np.sum((prop_q - prop_p) * np.log(prop_q / prop_p)))

        # Under H0, PSI is approximately (1/n + 1/m) * Chi2(B - 1). On small
        # samples that critical value is well above the usual 0.10 rule.
        df = max(1, b_actual - 1)
        scale = 1.0 / n + 1.0 / m
        psi_threshold = max(0.10, float(scale * stats.chi2.ppf(1.0 - alpha, df=df)))
        psi_pvalue = float(stats.chi2.sf(psi_val / scale, df=df))

        return psi_val, psi_threshold, psi_pvalue

    @staticmethod
    def effect_grade(psi_val: float, ks_stat: float, w_norm: float) -> str:
        """Size of a shift that has already been shown to be real."""
        if psi_val > 0.25 or ks_stat > 0.20 or w_norm > 0.40:
            return "SEVERE"
        if psi_val > 0.10 or ks_stat > 0.10 or w_norm > 0.20:
            return "MODERATE"
        return "LOW"

    @classmethod
    def analyze_feature(cls, feature_name: str, x_p: np.ndarray, x_q: np.ndarray) -> UnivariateDriftResult:
        """Run all univariate tests on a single feature and assign severity."""
        x_p_clean = x_p[np.isfinite(x_p)]
        x_q_clean = x_q[np.isfinite(x_q)]

        if len(x_p_clean) < 10 or len(x_q_clean) < 10:
            return UnivariateDriftResult(
                feature_name=feature_name,
                ks_statistic=0.0,
                ks_pvalue=1.0,
                ks_critical=1.0,
                wasserstein_norm=0.0,
                psi_score=0.0,
                psi_critical=0.10,
                p_value=1.0,
                is_drifted=False,
                severity="NONE",
            )

        ks_stat, ks_pval, ks_crit = cls.kolmogorov_smirnov(x_p_clean, x_q_clean)
        w_norm = cls.normalized_wasserstein(x_p_clean, x_q_clean)
        psi_val, psi_crit, psi_pval = cls.population_stability_index(x_p_clean, x_q_clean)

        # A shift has to be statistically real before its size is graded.
        # Grading effect size alone flags sampling noise on small samples.
        # Two tests per feature, so the smaller p-value is Bonferroni-doubled.
        p_value = min(1.0, 2.0 * min(ks_pval, psi_pval))
        drifted = p_value < 0.05
        sev = cls.effect_grade(psi_val, ks_stat, w_norm) if drifted else "NONE"

        return UnivariateDriftResult(
            feature_name=feature_name,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pval,
            ks_critical=ks_crit,
            wasserstein_norm=w_norm,
            psi_score=psi_val,
            psi_critical=psi_crit,
            p_value=p_value,
            is_drifted=drifted,
            severity=sev,
        )

    @classmethod
    def analyze_dataset(
        cls,
        df_reference: np.ndarray,
        df_target: np.ndarray,
        feature_names: List[str],
        q_fdr: float = 0.05,
    ) -> List[UnivariateDriftResult]:
        """Analyze all features with Benjamini-Hochberg False Discovery Rate control."""
        results: List[UnivariateDriftResult] = []
        p_values: List[float] = []

        for k, name in enumerate(feature_names):
            res = cls.analyze_feature(name, df_reference[:, k], df_target[:, k])
            results.append(res)
            p_values.append(res.p_value)

        # Benjamini-Hochberg: p_(i) <= (i / m) * q. Only features that pass
        # keep a severity grade; the rest are NONE whatever their effect size.
        n_features = len(p_values)
        sorted_p = np.sort(p_values)
        fdr_thresholds = (np.arange(1, n_features + 1) / n_features) * q_fdr
        passed = np.flatnonzero(sorted_p <= fdr_thresholds)
        cutoff = sorted_p[passed.max()] if passed.size else -1.0

        for res in results:
            significant = res.p_value <= cutoff
            res.is_drifted = bool(significant)
            res.severity = cls.effect_grade(res.psi_score, res.ks_statistic, res.wasserstein_norm) if significant else "NONE"

        return results
