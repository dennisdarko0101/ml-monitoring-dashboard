"""Metrics collection with sliding windows and Prometheus integration."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from prometheus_client import Counter, Gauge, Histogram, Summary

from src.storage.database import Database
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Prometheus metrics
PREDICTION_LATENCY = Histogram(
    "ml_prediction_latency_ms",
    "Prediction latency in milliseconds",
    ["model_name"],
    buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
)
PREDICTION_COUNT = Counter(
    "ml_prediction_total",
    "Total predictions",
    ["model_name"],
)
ERROR_COUNT = Counter(
    "ml_prediction_errors_total",
    "Total prediction errors",
    ["model_name"],
)
ACTIVE_MODELS = Gauge(
    "ml_active_models",
    "Number of actively monitored models",
)


@dataclass
class WindowEntry:
    """A single metric observation with timestamp."""

    value: float
    timestamp: float


@dataclass
class MetricsSummary:
    """Aggregated metrics summary for a model."""

    model_name: str
    window_hours: int
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    throughput_rps: float = 0.0
    error_rate: float = 0.0
    total_predictions: int = 0
    total_errors: int = 0
    prediction_distribution: dict[str, int] = field(default_factory=dict)


class SlidingWindow:
    """Time-based sliding window for metric aggregation."""

    def __init__(self, window_seconds: int) -> None:
        self.window_seconds = window_seconds
        self._entries: deque[WindowEntry] = deque()

    def add(self, value: float) -> None:
        now = time.time()
        self._entries.append(WindowEntry(value=value, timestamp=now))
        self._evict(now)

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._entries and self._entries[0].timestamp < cutoff:
            self._entries.popleft()

    def values(self) -> list[float]:
        self._evict(time.time())
        return [e.value for e in self._entries]

    def count(self) -> int:
        self._evict(time.time())
        return len(self._entries)

    def is_empty(self) -> bool:
        return self.count() == 0


class MetricsCollector:
    """Collects and aggregates model performance metrics with sliding windows."""

    WINDOWS = {
        "1h": 3600,
        "24h": 86400,
        "7d": 604800,
    }

    def __init__(self, db: Database | None = None) -> None:
        self.db = db
        # model_name -> metric_name -> window_name -> SlidingWindow
        self._windows: dict[str, dict[str, dict[str, SlidingWindow]]] = {}
        self._prediction_counts: dict[str, int] = {}
        self._error_counts: dict[str, int] = {}
        self._prediction_labels: dict[str, dict[str, int]] = {}

    def _ensure_windows(self, model_name: str, metric_name: str) -> None:
        if model_name not in self._windows:
            self._windows[model_name] = {}
            self._prediction_counts[model_name] = 0
            self._error_counts[model_name] = 0
            self._prediction_labels[model_name] = {}
        if metric_name not in self._windows[model_name]:
            self._windows[model_name][metric_name] = {
                name: SlidingWindow(secs) for name, secs in self.WINDOWS.items()
            }

    def record(self, model_name: str, metric_name: str, value: float) -> None:
        """Record a metric value into all sliding windows."""
        self._ensure_windows(model_name, metric_name)
        for window in self._windows[model_name][metric_name].values():
            window.add(value)

        # Update Prometheus
        if metric_name == "latency":
            PREDICTION_LATENCY.labels(model_name=model_name).observe(value)
        elif metric_name == "error":
            ERROR_COUNT.labels(model_name=model_name).inc(value)

    def record_prediction(
        self, model_name: str, latency_ms: float, prediction_label: str | None = None,
        is_error: bool = False,
    ) -> None:
        """Convenience: record a full prediction event."""
        self._ensure_windows(model_name, "latency")
        self.record(model_name, "latency", latency_ms)

        self._prediction_counts.setdefault(model_name, 0)
        self._prediction_counts[model_name] += 1
        PREDICTION_COUNT.labels(model_name=model_name).inc()

        if is_error:
            self._error_counts.setdefault(model_name, 0)
            self._error_counts[model_name] += 1
            ERROR_COUNT.labels(model_name=model_name).inc()

        if prediction_label is not None:
            self._prediction_labels.setdefault(model_name, {})
            self._prediction_labels[model_name].setdefault(prediction_label, 0)
            self._prediction_labels[model_name][prediction_label] += 1

        ACTIVE_MODELS.set(len(self._prediction_counts))

    def get_summary(self, model_name: str, window: str = "1h") -> MetricsSummary:
        """Get aggregated metrics for a model over a time window."""
        latencies = self._get_values(model_name, "latency", window)
        total_preds = self._prediction_counts.get(model_name, 0)
        total_errs = self._error_counts.get(model_name, 0)

        if latencies:
            arr = np.array(latencies)
            p50 = float(np.percentile(arr, 50))
            p95 = float(np.percentile(arr, 95))
            p99 = float(np.percentile(arr, 99))
        else:
            p50 = p95 = p99 = 0.0

        window_secs = self.WINDOWS.get(window, 3600)
        throughput = len(latencies) / window_secs if latencies else 0.0
        error_rate = total_errs / total_preds if total_preds > 0 else 0.0

        return MetricsSummary(
            model_name=model_name,
            window_hours=window_secs // 3600,
            latency_p50=round(p50, 2),
            latency_p95=round(p95, 2),
            latency_p99=round(p99, 2),
            throughput_rps=round(throughput, 4),
            error_rate=round(error_rate, 4),
            total_predictions=total_preds,
            total_errors=total_errs,
            prediction_distribution=dict(self._prediction_labels.get(model_name, {})),
        )

    def _get_values(self, model_name: str, metric_name: str, window: str) -> list[float]:
        try:
            return self._windows[model_name][metric_name][window].values()
        except KeyError:
            return []

    async def persist_snapshot(self, model_name: str) -> None:
        """Save current metrics to the database for historical analysis."""
        if not self.db:
            return
        summary = self.get_summary(model_name, "1h")
        for metric_name, value in [
            ("latency_p50", summary.latency_p50),
            ("latency_p95", summary.latency_p95),
            ("latency_p99", summary.latency_p99),
            ("throughput_rps", summary.throughput_rps),
            ("error_rate", summary.error_rate),
        ]:
            await self.db.insert_metric(model_name, metric_name, value)
