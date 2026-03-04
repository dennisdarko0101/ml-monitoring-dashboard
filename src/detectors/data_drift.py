"""Statistical data drift detection for numeric and categorical features."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from src.config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureDriftResult:
    """Drift result for a single feature."""

    feature: str
    drift_score: float
    method: str
    p_value: float | None = None
    is_drifted: bool = False
    details: dict = field(default_factory=dict)


@dataclass
class DriftReport:
    """Overall drift detection report."""

    per_feature_scores: list[FeatureDriftResult]
    overall_drift_score: float
    drifted_features: list[str]
    is_drifted: bool
    timestamp: str = ""


class DataDriftDetector:
    """Detects data drift between reference and current distributions."""

    def __init__(
        self,
        ks_threshold: float | None = None,
        psi_threshold: float | None = None,
        js_threshold: float | None = None,
    ) -> None:
        self.ks_threshold = ks_threshold or settings.DRIFT_THRESHOLD
        self.psi_threshold = psi_threshold or settings.PSI_THRESHOLD
        self.js_threshold = js_threshold or settings.JS_DIVERGENCE_THRESHOLD
        self._reference: pd.DataFrame | None = None

    def set_reference(self, reference_data: pd.DataFrame) -> None:
        """Set the reference (baseline) distribution."""
        self._reference = reference_data.copy()

    def detect(self, current_data: pd.DataFrame) -> DriftReport:
        """Run drift detection comparing current data against reference."""
        if self._reference is None:
            raise ValueError("Reference data not set. Call set_reference() first.")

        results: list[FeatureDriftResult] = []
        common_cols = [c for c in self._reference.columns if c in current_data.columns]

        for col in common_cols:
            ref_col = self._reference[col].dropna()
            cur_col = current_data[col].dropna()

            if ref_col.empty or cur_col.empty:
                continue

            if pd.api.types.is_numeric_dtype(ref_col):
                result = self._detect_numeric(col, ref_col, cur_col)
            else:
                result = self._detect_categorical(col, ref_col, cur_col)
            results.append(result)

        drifted = [r.feature for r in results if r.is_drifted]
        overall_score = (
            float(np.mean([r.drift_score for r in results])) if results else 0.0
        )

        return DriftReport(
            per_feature_scores=results,
            overall_drift_score=round(overall_score, 4),
            drifted_features=drifted,
            is_drifted=len(drifted) > 0,
        )

    def _detect_numeric(
        self, feature: str, ref: pd.Series, cur: pd.Series
    ) -> FeatureDriftResult:
        """Detect drift in numeric features using KS test, PSI, Wasserstein."""
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.ks_2samp(ref.values, cur.values)

        # Population Stability Index
        psi = self._compute_psi(ref.values, cur.values)

        # Wasserstein distance (Earth Mover's Distance)
        wasserstein = float(stats.wasserstein_distance(ref.values, cur.values))

        # Normalize Wasserstein by reference std for comparability
        ref_std = float(ref.std()) if float(ref.std()) > 0 else 1.0
        wasserstein_norm = wasserstein / ref_std

        is_drifted = ks_p < self.ks_threshold or psi > self.psi_threshold

        return FeatureDriftResult(
            feature=feature,
            drift_score=round(float(ks_stat), 4),
            method="ks_test+psi",
            p_value=round(float(ks_p), 6),
            is_drifted=is_drifted,
            details={
                "ks_statistic": round(float(ks_stat), 4),
                "ks_p_value": round(float(ks_p), 6),
                "psi": round(psi, 4),
                "wasserstein": round(wasserstein, 4),
                "wasserstein_normalized": round(wasserstein_norm, 4),
            },
        )

    def _detect_categorical(
        self, feature: str, ref: pd.Series, cur: pd.Series
    ) -> FeatureDriftResult:
        """Detect drift in categorical features using chi-squared and JS divergence."""
        all_categories = sorted(set(ref.unique()) | set(cur.unique()))
        ref_counts = ref.value_counts()
        cur_counts = cur.value_counts()

        ref_freq = np.array([ref_counts.get(c, 0) for c in all_categories], dtype=float)
        cur_freq = np.array([cur_counts.get(c, 0) for c in all_categories], dtype=float)

        # Chi-squared test
        # Add small constant to avoid zero-division
        ref_expected = ref_freq / ref_freq.sum() * cur_freq.sum()
        ref_expected = np.maximum(ref_expected, 1e-10)
        chi2_stat, chi2_p = stats.chisquare(cur_freq + 1e-10, f_exp=ref_expected + 1e-10)

        # Jensen-Shannon divergence
        js_div = self._jensen_shannon(ref_freq, cur_freq)

        is_drifted = chi2_p < self.ks_threshold or js_div > self.js_threshold

        return FeatureDriftResult(
            feature=feature,
            drift_score=round(js_div, 4),
            method="chi_squared+js_divergence",
            p_value=round(float(chi2_p), 6),
            is_drifted=is_drifted,
            details={
                "chi2_statistic": round(float(chi2_stat), 4),
                "chi2_p_value": round(float(chi2_p), 6),
                "js_divergence": round(js_div, 4),
            },
        )

    @staticmethod
    def _compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Compute Population Stability Index."""
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        if min_val == max_val:
            return 0.0

        bin_edges = np.linspace(min_val, max_val, bins + 1)

        ref_hist, _ = np.histogram(reference, bins=bin_edges)
        cur_hist, _ = np.histogram(current, bins=bin_edges)

        # Convert to proportions with smoothing
        ref_prop = (ref_hist + 1e-6) / (ref_hist.sum() + bins * 1e-6)
        cur_prop = (cur_hist + 1e-6) / (cur_hist.sum() + bins * 1e-6)

        psi = float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))
        return max(psi, 0.0)

    @staticmethod
    def _jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
        """Compute Jensen-Shannon divergence between two distributions."""
        p_norm = p / (p.sum() + 1e-10)
        q_norm = q / (q.sum() + 1e-10)
        m = 0.5 * (p_norm + q_norm)

        # KL divergence with smoothing
        def _kl(a: np.ndarray, b: np.ndarray) -> float:
            mask = a > 0
            return float(np.sum(a[mask] * np.log(a[mask] / (b[mask] + 1e-10))))

        return 0.5 * _kl(p_norm, m) + 0.5 * _kl(q_norm, m)
