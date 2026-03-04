"""Anomaly detection for latency, error rates, and request volume."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """A detected anomaly in a metric stream."""

    metric: str
    value: float
    expected_range: tuple[float, float]
    severity: Severity
    timestamp: float
    method: str


class AnomalyDetector:
    """Detects anomalies in operational metrics using statistical methods."""

    def __init__(
        self,
        z_score_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        rolling_window: int = 50,
    ) -> None:
        self.z_score_threshold = z_score_threshold
        self.iqr_multiplier = iqr_multiplier
        self.rolling_window = rolling_window

    def detect_latency_anomalies(
        self, latencies: Sequence[float], timestamps: Sequence[float] | None = None
    ) -> list[Anomaly]:
        """Detect anomalous latency values using z-score and IQR methods."""
        if len(latencies) < 5:
            return []

        arr = np.array(latencies)
        ts = np.array(timestamps) if timestamps else np.arange(len(arr), dtype=float)
        anomalies: list[Anomaly] = []

        # Z-score method
        z_anomalies = self._z_score_detect(arr, ts, "latency")
        anomalies.extend(z_anomalies)

        # IQR method
        iqr_anomalies = self._iqr_detect(arr, ts, "latency")
        anomalies.extend(iqr_anomalies)

        return self._deduplicate(anomalies)

    def detect_error_spikes(
        self, error_counts: Sequence[float], timestamps: Sequence[float] | None = None
    ) -> list[Anomaly]:
        """Detect sudden spikes in error rates."""
        if len(error_counts) < 5:
            return []

        arr = np.array(error_counts, dtype=float)
        ts = np.array(timestamps) if timestamps else np.arange(len(arr), dtype=float)
        anomalies: list[Anomaly] = []

        # Rolling average deviation
        rolling = self._rolling_deviation_detect(arr, ts, "error_rate")
        anomalies.extend(rolling)

        # Z-score for spike detection
        z_anomalies = self._z_score_detect(arr, ts, "error_rate")
        anomalies.extend(z_anomalies)

        return self._deduplicate(anomalies)

    def detect_volume_anomalies(
        self, request_counts: Sequence[float], timestamps: Sequence[float] | None = None
    ) -> list[Anomaly]:
        """Detect anomalies in request volume (both drops and surges)."""
        if len(request_counts) < 5:
            return []

        arr = np.array(request_counts, dtype=float)
        ts = np.array(timestamps) if timestamps else np.arange(len(arr), dtype=float)
        anomalies: list[Anomaly] = []

        # IQR detects both high and low outliers
        iqr_anomalies = self._iqr_detect(arr, ts, "request_volume")
        anomalies.extend(iqr_anomalies)

        # Rolling average for trend deviation
        rolling = self._rolling_deviation_detect(arr, ts, "request_volume")
        anomalies.extend(rolling)

        return self._deduplicate(anomalies)

    def _z_score_detect(
        self, data: np.ndarray, timestamps: np.ndarray, metric: str
    ) -> list[Anomaly]:
        """Flag points whose z-score exceeds the threshold."""
        mean = float(data.mean())
        std = float(data.std())
        if std < 1e-10:
            return []

        anomalies = []
        for i, val in enumerate(data):
            z = abs((val - mean) / std)
            if z > self.z_score_threshold:
                low = mean - self.z_score_threshold * std
                high = mean + self.z_score_threshold * std
                severity = Severity.CRITICAL if z > self.z_score_threshold * 1.5 else Severity.WARNING
                anomalies.append(
                    Anomaly(
                        metric=metric,
                        value=float(val),
                        expected_range=(round(low, 2), round(high, 2)),
                        severity=severity,
                        timestamp=float(timestamps[i]),
                        method="z_score",
                    )
                )
        return anomalies

    def _iqr_detect(
        self, data: np.ndarray, timestamps: np.ndarray, metric: str
    ) -> list[Anomaly]:
        """Flag points outside IQR fences."""
        q1 = float(np.percentile(data, 25))
        q3 = float(np.percentile(data, 75))
        iqr = q3 - q1
        if iqr < 1e-10:
            return []

        lower = q1 - self.iqr_multiplier * iqr
        upper = q3 + self.iqr_multiplier * iqr

        anomalies = []
        for i, val in enumerate(data):
            if val < lower or val > upper:
                severity = (
                    Severity.CRITICAL
                    if val > upper + iqr or val < lower - iqr
                    else Severity.WARNING
                )
                anomalies.append(
                    Anomaly(
                        metric=metric,
                        value=float(val),
                        expected_range=(round(lower, 2), round(upper, 2)),
                        severity=severity,
                        timestamp=float(timestamps[i]),
                        method="iqr",
                    )
                )
        return anomalies

    def _rolling_deviation_detect(
        self, data: np.ndarray, timestamps: np.ndarray, metric: str
    ) -> list[Anomaly]:
        """Detect deviations from a rolling average."""
        window = min(self.rolling_window, len(data) // 2)
        if window < 3:
            return []

        anomalies = []
        for i in range(window, len(data)):
            window_data = data[i - window : i]
            rolling_mean = float(window_data.mean())
            rolling_std = float(window_data.std())
            if rolling_std < 1e-10:
                continue

            deviation = abs(data[i] - rolling_mean) / rolling_std
            if deviation > self.z_score_threshold:
                low = rolling_mean - self.z_score_threshold * rolling_std
                high = rolling_mean + self.z_score_threshold * rolling_std
                anomalies.append(
                    Anomaly(
                        metric=metric,
                        value=float(data[i]),
                        expected_range=(round(low, 2), round(high, 2)),
                        severity=Severity.WARNING,
                        timestamp=float(timestamps[i]),
                        method="rolling_deviation",
                    )
                )
        return anomalies

    @staticmethod
    def _deduplicate(anomalies: list[Anomaly]) -> list[Anomaly]:
        """Remove duplicate anomalies at the same timestamp/metric."""
        seen: set[tuple[str, float]] = set()
        unique = []
        for a in anomalies:
            key = (a.metric, a.timestamp)
            if key not in seen:
                seen.add(key)
                unique.append(a)
        return unique
