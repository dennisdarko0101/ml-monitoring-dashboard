"""Tests for MetricsCollector: sliding windows, percentiles, Prometheus."""

from __future__ import annotations

import time

import pytest

from src.collectors.metrics_collector import MetricsCollector, MetricsSummary, SlidingWindow


class TestSlidingWindow:
    def test_add_and_values(self):
        w = SlidingWindow(3600)
        w.add(1.0)
        w.add(2.0)
        w.add(3.0)
        assert len(w.values()) == 3

    def test_count(self):
        w = SlidingWindow(3600)
        for i in range(10):
            w.add(float(i))
        assert w.count() == 10

    def test_is_empty(self):
        w = SlidingWindow(3600)
        assert w.is_empty()
        w.add(1.0)
        assert not w.is_empty()


class TestMetricsCollector:
    def test_record_creates_windows(self, metrics_collector):
        metrics_collector.record("model_a", "latency", 50.0)
        vals = metrics_collector._get_values("model_a", "latency", "1h")
        assert len(vals) == 1
        assert vals[0] == 50.0

    def test_record_prediction(self, metrics_collector):
        metrics_collector.record_prediction("model_a", 50.0, "positive")
        metrics_collector.record_prediction("model_a", 60.0, "negative")
        assert metrics_collector._prediction_counts["model_a"] == 2

    def test_record_prediction_with_error(self, metrics_collector):
        metrics_collector.record_prediction("model_a", 50.0, is_error=True)
        assert metrics_collector._error_counts["model_a"] == 1

    def test_get_summary_empty(self, metrics_collector):
        summary = metrics_collector.get_summary("nonexistent", "1h")
        assert summary.latency_p50 == 0.0
        assert summary.total_predictions == 0

    def test_get_summary_with_data(self, metrics_collector):
        for i in range(100):
            metrics_collector.record_prediction("model_a", float(i), "label")
        summary = metrics_collector.get_summary("model_a", "1h")
        assert summary.latency_p50 > 0
        assert summary.latency_p95 > summary.latency_p50
        assert summary.latency_p99 >= summary.latency_p95
        assert summary.total_predictions == 100

    def test_get_summary_error_rate(self, metrics_collector):
        for i in range(10):
            is_err = i < 3  # 30% errors
            metrics_collector.record_prediction("model_a", 50.0, is_error=is_err)
        summary = metrics_collector.get_summary("model_a", "1h")
        assert summary.error_rate == pytest.approx(0.3, abs=0.01)

    def test_prediction_distribution(self, metrics_collector):
        metrics_collector.record_prediction("m", 10.0, "cat")
        metrics_collector.record_prediction("m", 10.0, "cat")
        metrics_collector.record_prediction("m", 10.0, "dog")
        summary = metrics_collector.get_summary("m", "1h")
        assert summary.prediction_distribution == {"cat": 2, "dog": 1}

    def test_multiple_models(self, metrics_collector):
        metrics_collector.record_prediction("model_a", 10.0)
        metrics_collector.record_prediction("model_b", 20.0)
        sa = metrics_collector.get_summary("model_a", "1h")
        sb = metrics_collector.get_summary("model_b", "1h")
        assert sa.total_predictions == 1
        assert sb.total_predictions == 1

    def test_throughput_calculation(self, metrics_collector):
        for _ in range(100):
            metrics_collector.record_prediction("m", 10.0)
        summary = metrics_collector.get_summary("m", "1h")
        # 100 requests in a 3600-second window
        assert summary.throughput_rps > 0

    async def test_persist_snapshot(self, db, metrics_collector):
        for i in range(50):
            metrics_collector.record_prediction("m", float(i))
        await metrics_collector.persist_snapshot("m")
        metrics = await db.get_metrics("m")
        assert len(metrics) > 0

    def test_record_multiple_metrics(self, metrics_collector):
        metrics_collector.record("m", "latency", 10.0)
        metrics_collector.record("m", "latency", 20.0)
        metrics_collector.record("m", "custom_metric", 5.0)
        lat_vals = metrics_collector._get_values("m", "latency", "1h")
        custom_vals = metrics_collector._get_values("m", "custom_metric", "1h")
        assert len(lat_vals) == 2
        assert len(custom_vals) == 1
