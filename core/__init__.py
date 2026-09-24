"""driftdetect: Covariate Shift Quantification, Adversarial Validation, and Train-Test Alignment."""
from core.stats import UnivariateDriftAnalyzer, UnivariateDriftResult
from core.adversarial import AdversarialValidator, AdversarialValidationResult
from core.attribution import DriftAttributionEngine, CulpritFeature
from core.alignment import ImportanceWeightCalculator, AdversarialHoldoutSplitter, AdversarialStratifiedKFold
from core.visualizer import DriftVisualizer

__all__ = [
    "UnivariateDriftAnalyzer",
    "UnivariateDriftResult",
    "AdversarialValidator",
    "AdversarialValidationResult",
    "DriftAttributionEngine",
    "CulpritFeature",
    "ImportanceWeightCalculator",
    "AdversarialHoldoutSplitter",
    "AdversarialStratifiedKFold",
    "DriftVisualizer",
]
