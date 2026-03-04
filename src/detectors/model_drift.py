"""Model-level drift: concept drift, prediction drift, performance degradation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelDriftReport:
    """Report for model-level drift detection."""

    concept_drift_detected: bool
    prediction_drift_detected: bool
    performance_degradation_detected: bool
    accuracy_trend: float  # slope of accuracy over time, negative = degrading
    prediction_distribution_shift: float
    change_points: list[int]
    details: dict = field(default_factory=dict)

    @property
    def any_drift(self) -> bool:
        return (
            self.concept_drift_detected
            or self.prediction_drift_detected
            or self.performance_degradation_detected
        )


class PageHinkley:
    """Page-Hinkley change detection algorithm."""

    def __init__(
        self,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 1 - 0.0001,
    ) -> None:
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self._sum = 0.0
        self._count = 0
        self._mean = 0.0
        self._min_sum = float("inf")

    def update(self, value: float) -> bool:
        """Add a new observation. Returns True if change detected."""
        self._count += 1
        self._mean = self._mean * self.alpha + value * (1 - self.alpha)
        self._sum += value - self._mean - self.delta
        self._min_sum = min(self._min_sum, self._sum)
        return (self._sum - self._min_sum) > self.threshold

    def reset(self) -> None:
        self._sum = 0.0
        self._count = 0
        self._mean = 0.0
        self._min_sum = float("inf")


class ADWIN:
    """Simplified ADWIN (Adaptive Windowing) for change detection."""

    def __init__(self, delta: float = 0.002) -> None:
        self.delta = delta
        self._window: list[float] = []
        self._max_window = 2000

    def update(self, value: float) -> bool:
        """Add observation, detect if a change has occurred."""
        self._window.append(value)
        if len(self._window) > self._max_window:
            self._window = self._window[-self._max_window :]

        if len(self._window) < 20:
            return False

        return self._check_split()

    def _check_split(self) -> bool:
        """Check if splitting the window reveals a distributional change."""
        n = len(self._window)
        # Test a few split points
        for split_frac in [0.3, 0.5, 0.7]:
            split = int(n * split_frac)
            if split < 5 or (n - split) < 5:
                continue
            left = np.array(self._window[:split])
            right = np.array(self._window[split:])

            mean_diff = abs(float(left.mean()) - float(right.mean()))
            pooled_var = (float(left.var()) + float(right.var())) / 2
            if pooled_var < 1e-10:
                continue

            # Hoeffding-style bound
            m = 1.0 / split + 1.0 / (n - split)
            bound = np.sqrt(2.0 * m * pooled_var * np.log(2.0 / self.delta))
            if mean_diff > bound:
                return True
        return False

    def reset(self) -> None:
        self._window.clear()


class ModelDriftDetector:
    """Detects concept drift, prediction drift, and performance degradation."""

    def __init__(
        self,
        accuracy_window: int = 100,
        degradation_threshold: float = -0.01,
        ph_threshold: float = 50.0,
    ) -> None:
        self.accuracy_window = accuracy_window
        self.degradation_threshold = degradation_threshold
        self._ph_detector = PageHinkley(threshold=ph_threshold)
        self._adwin = ADWIN()

    def detect(
        self,
        recent_predictions: np.ndarray,
        recent_actuals: np.ndarray,
        baseline_predictions: np.ndarray | None = None,
    ) -> ModelDriftReport:
        """Run all model drift checks."""
        # Concept drift: monitor accuracy changes
        concept_drift, change_points, accuracy_trend = self._detect_concept_drift(
            recent_predictions, recent_actuals
        )

        # Prediction drift: shift in prediction distribution
        pred_drift, pred_shift = self._detect_prediction_drift(
            recent_predictions, baseline_predictions
        )

        # Performance degradation: accuracy declining over time
        perf_degradation = accuracy_trend < self.degradation_threshold

        return ModelDriftReport(
            concept_drift_detected=concept_drift,
            prediction_drift_detected=pred_drift,
            performance_degradation_detected=perf_degradation,
            accuracy_trend=round(accuracy_trend, 6),
            prediction_distribution_shift=round(pred_shift, 4),
            change_points=change_points,
            details={
                "accuracy_window": self.accuracy_window,
                "n_predictions": len(recent_predictions),
                "n_actuals": len(recent_actuals),
            },
        )

    def _detect_concept_drift(
        self, predictions: np.ndarray, actuals: np.ndarray
    ) -> tuple[bool, list[int], float]:
        """Detect concept drift via accuracy monitoring with Page-Hinkley."""
        if len(predictions) == 0 or len(actuals) == 0:
            return False, [], 0.0

        n = min(len(predictions), len(actuals))
        correct = (predictions[:n] == actuals[:n]).astype(float)

        # Page-Hinkley on error signal
        self._ph_detector.reset()
        change_points = []
        for i, c in enumerate(correct):
            error = 1.0 - c
            if self._ph_detector.update(error):
                change_points.append(i)
                self._ph_detector.reset()

        # Accuracy trend (linear regression slope)
        if len(correct) >= 10:
            window_size = min(self.accuracy_window, len(correct))
            windowed_acc = []
            step = max(1, window_size // 10)
            for i in range(0, len(correct) - window_size + 1, step):
                windowed_acc.append(float(correct[i : i + window_size].mean()))
            if len(windowed_acc) >= 2:
                x = np.arange(len(windowed_acc))
                slope, _, _, _, _ = stats.linregress(x, windowed_acc)
                accuracy_trend = float(slope)
            else:
                accuracy_trend = 0.0
        else:
            accuracy_trend = 0.0

        concept_drift = len(change_points) > 0
        return concept_drift, change_points, accuracy_trend

    def _detect_prediction_drift(
        self,
        current_predictions: np.ndarray,
        baseline_predictions: np.ndarray | None,
    ) -> tuple[bool, float]:
        """Detect shift in prediction distribution."""
        if baseline_predictions is None or len(baseline_predictions) == 0:
            return False, 0.0

        if len(current_predictions) == 0:
            return False, 0.0

        # Use KS test for numeric predictions
        try:
            cur_float = current_predictions.astype(float)
            base_float = baseline_predictions.astype(float)
            ks_stat, ks_p = stats.ks_2samp(cur_float, base_float)
            is_drifted = ks_p < 0.05
            return is_drifted, float(ks_stat)
        except (ValueError, TypeError):
            # Categorical: compare distributions via chi-squared
            all_labels = sorted(set(current_predictions) | set(baseline_predictions))
            cur_counts = np.array(
                [np.sum(current_predictions == l) for l in all_labels], dtype=float
            )
            base_counts = np.array(
                [np.sum(baseline_predictions == l) for l in all_labels], dtype=float
            )
            base_expected = base_counts / (base_counts.sum() + 1e-10) * cur_counts.sum()
            base_expected = np.maximum(base_expected, 1e-10)
            chi2, p = stats.chisquare(cur_counts + 1e-10, f_exp=base_expected)
            return p < 0.05, float(chi2)
