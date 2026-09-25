# driftdetect

Production covariate shift quantification, adversarial validation, and adaptive train-test alignment engine for competitive machine learning and tabular production pipelines.

![Drift Diagnostic Dashboard](samples/drift_dashboard.png)

*Comprehensive drift diagnostics: Out-of-fold adversarial validation ROC curve, population stability index (PSI) per feature, empirical CDF divergence for primary culprit features, and out-of-fold probability density separation.*

---

## Covariate Shift Detection & Validation Strategy

In competitive machine learning and production scoring systems, model performance deteriorates when test feature distributions shift relative to training data:

$$P_{\text{train}}(X) \neq P_{\text{test}}(X)$$

Addressing distribution shift requires separating benign sampling noise from genuine structural drift:
1. **Sample Size vs Effect Size:** On large sample sizes ($N > 10^4$), standard null-hypothesis p-values flag almost all features. `driftdetect` couples p-values with standardized effect size metrics (normalized $W_1$, calibrated PSI).
2. **Multivariate Dependencies:** Features may show identical univariate marginals while shifting significantly in their joint correlation structure.
3. **Data Leakage Screening:** Monotonic IDs or timestamps that artificially separate train from test are identified and excluded before model validation.

```mermaid
flowchart TD
    subgraph Ingestion ["Data Ingestion & Filtering"]
        Train["Train Reference (P)"] --> Guard["Leakage Guard (AUC >= 0.98 Exclusion)"]
        Test["Test Target (Q)"] --> Guard
    end

    Guard --> Univar["1. Univariate Statistics\nKS-Test, Normalized W1, Calibrated PSI, BH-FDR"]
    Guard --> Adv["2. Adversarial Validation\n5-Fold OOF Discriminator, ROC-AUC + 95% CI"]

    Adv -->|If AUC > 0.55| Attrib["3. Drift Attribution & Elimination\nPermutation Importance, Consensus C_k, RAFE"]
    Adv --> Align["4. Adaptive Alignment\nImportance-Weighted CV (IWCV), Adversarial Splitters"]
```

---

## Mathematical Formulations

### 1. Sample-Size Aware Population Stability Index (PSI)
Standard industrial rules use fixed thresholds ($\text{PSI} < 0.10$). On small sample sizes ($N < 500$), random sampling noise triggers false alarms. Under the null hypothesis $H_0: P = Q$, the empirical PSI follows a scaled Chi-square distribution:

$$\text{PSI} \sim \left(\frac{1}{n} + \frac{1}{m}\right) \chi^2_{B-1}$$

`core/stats.py` computes sample-size calibrated thresholds:

$$\text{PSI}_{\text{crit}}(\alpha) = \max\left(0.10, \; \left(\frac{1}{n} + \frac{1}{m}\right) \chi^2_{B-1, \, 1-\alpha}\right)$$

### 2. Normalized Wasserstein-1 Distance
Univariate Earth Mover's Distance measures total integral mass discrepancy between empirical CDFs:

$$W_1(P, Q) = \int_{-\infty}^\infty |F_P(x) - F_Q(x)| \, dx$$

To enable comparisons across features with wildly different units (e.g. `age` vs `monthly_income`), $W_1$ is normalized by the interquartile range of the reference distribution:

$$\tilde{W}_1 = \frac{W_1(P, Q)}{\text{IQR}(X_P) + \epsilon}$$

### 3. Recursive Adversarial Feature Elimination (RAFE)
When adversarial validation reveals severe separation ($\text{AUC} > 0.80$), RAFE iteratively eliminates the top culprit feature and measures the decay of discriminability:

![RAFE Decay](samples/rafe_decay.png)

```
Iteration 0 (Baseline All Features) : Adversarial AUC = 0.8322 [Severe Shift]
Iteration 1 (Drop debt_to_income)   : Adversarial AUC = 0.6614
Iteration 2 (Drop user_age)         : Adversarial AUC = 0.5794
Iteration 3 (Drop monthly_income)   : Adversarial AUC = 0.4996 [Perfect Alignment]
```

### 4. Density Ratio & Importance-Weighted Cross-Validation (IWCV)
Using the out-of-fold probability $s(x) = P(\text{test} \mid x)$, Bayes' rule yields the Radon-Nikodym density derivative:

$$w(x) = \frac{q(x)}{p(x)} = \frac{n}{m} \frac{s(x)}{1 - s(x)}$$

To prevent high-variance weights from destabilizing gradient boosting estimators, weights are stabilized via Shimodaira power-flattening ($\lambda = 0.5$) and 99th-percentile clipping:

$$w_i \leftarrow \min\left(w_i^\lambda, \; Q_{0.99}(w)\right) \cdot \frac{n}{\sum w_i}$$

---

## Audit Output Example

| Feature | KS Stat | p-val | W1 Norm | PSI | Severity Assessment |
|---|---|---|---|---|---|
| `debt_to_income_ratio` | 0.471 | 2.84e-201 | 1.086 | 1.378 | **SEVERE (Primary Culprit)** |
| `user_age` | 0.240 | 5.61e-51 | 0.456 | 0.316 | **SEVERE (Shifted Mean)** |
| `monthly_income` | 0.187 | 6.92e-31 | 0.529 | 0.195 | **MODERATE / SEVERE** |
| `interest_rate` | 0.038 | 0.110 | 0.044 | 0.012 | NONE (Stable) |
| `revolving_utilization` | 0.031 | 0.290 | 0.018 | 0.012 | NONE (Stable) |
| `credit_lines_count` | 0.022 | 0.698 | 0.023 | 0.009 | NONE (Stable) |
| `loan_amount` | 0.021 | 0.750 | 0.019 | 0.006 | NONE (Stable) |
| `inquiry_count_6m` | 0.013 | 0.994 | 0.019 | 0.002 | NONE (Stable) |
| `delinquency_history` | 0.001 | 1.000 | 0.003 | 0.000 | NONE (Stable) |
| `transaction_id` | 1.000 | 0.000 | 1.501 | 14.87 | **LEAKAGE (Excluded)** |

- **Leakage Guard Action:** `transaction_id` identified as monotonic separator ($\text{AUC} = 1.0$) and excluded prior to adversarial validation.
- **Out-of-Fold ROC-AUC:** 0.8301 [95% CI: 0.8166 - 0.8424] (Gini = 0.6602).
- **Effective Sample Size:** 58.8% of training set under stabilized importance weighting ($\lambda = 0.5$).

---

## Project Structure

```
driftdetect/
├── core/
│   ├── stats.py           # Kolmogorov-Smirnov, Normalized W1, Calibrated PSI, and BH-FDR
│   ├── adversarial.py     # Leakage pre-pass guard, LightGBM/RF out-of-fold validation, and bootstrap CI
│   ├── attribution.py     # Permutation importance, consensus culprit score C_k, and RAFE trajectory
│   ├── alignment.py       # Shimodaira importance weighting, AdversarialHoldout, AdversarialStratifiedKFold
│   └── visualizer.py      # Multi-panel dashboard (ROC, PSI bars, ECDF shift, density histograms)
├── samples/
│   ├── drift_dashboard.png # Comprehensive 4-panel diagnostic plot
│   └── rafe_decay.png      # Feature elimination decay trajectory
├── scripts/
│   └── run_audit.py        # End-to-end audit demonstration script
├── tests/
│   ├── test_stats.py       # KS, W1, and PSI invariants
│   ├── test_adversarial.py # Adversarial validation AUC on null and drifted distributions
│   ├── test_attribution.py # Culprit identification ranking
│   └── test_alignment.py   # Importance weights and cross-validation split fractions
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/muqsithanif/driftdetect.git
cd driftdetect

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Audit

```bash
python scripts/run_audit.py
```
Performs leakage detection, univariate statistical testing, 5-fold adversarial validation, culprit attribution, and exports diagnostic dashboards to `samples/`.

### 3. Run Automated Tests

```bash
pytest tests -v
```

---

## Automated Invariant Tests

Eight unit tests enforce statistical, adversarial, and alignment guarantees:
- **`test_stats.py`:** Enforces null invariance ($D < D_{\text{crit}}$ and $\text{PSI} < 0.10$ on identical draws), validates severe shift detection, and proves scale invariance of normalized Wasserstein distance.
- **`test_adversarial.py`:** Asserts $\text{AUC} \approx 0.50$ on null distributions and verifies automated identification of monotonic index leakage.
- **`test_attribution.py`:** Proves that deliberately shifted features are ranked as primary culprits ($C_k \ge 0.70$).
- **`test_alignment.py`:** Confirms importance weights sum to $N_{\text{train}}$, verifies positive bounded Effective Sample Size, and tests holdout split fraction integrity.
