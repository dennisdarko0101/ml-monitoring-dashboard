"""Tests for ChartGenerator: each chart type produces valid Plotly JSON."""

from __future__ import annotations

import pytest

from src.dashboard.charts import ChartGenerator


class TestChartGenerator:
    def test_latency_timeseries(self):
        result = ChartGenerator.latency_timeseries(
            timestamps=["2024-01-01", "2024-01-02", "2024-01-03"],
            p50=[10.0, 12.0, 11.0],
            p95=[50.0, 55.0, 52.0],
            p99=[100.0, 110.0, 105.0],
        )
        assert "data" in result
        assert len(result["data"]) == 3  # P50, P95, P99 traces

    def test_throughput_chart(self):
        result = ChartGenerator.throughput_chart(
            timestamps=["2024-01-01", "2024-01-02"],
            rps=[100.0, 120.0],
        )
        assert "data" in result
        assert len(result["data"]) >= 1

    def test_error_rate_chart(self):
        result = ChartGenerator.error_rate_chart(
            timestamps=["2024-01-01", "2024-01-02"],
            error_rates=[0.01, 0.02],
            threshold=0.05,
        )
        assert "data" in result
        assert "layout" in result

    def test_drift_heatmap(self):
        result = ChartGenerator.drift_heatmap(
            features=["feat_a", "feat_b", "feat_c"],
            drift_scores=[0.1, 0.8, 0.3],
        )
        assert "data" in result
        assert result["data"][0]["type"] == "heatmap"

    def test_prediction_distribution(self):
        result = ChartGenerator.prediction_distribution(
            labels=["positive", "negative", "neutral"],
            counts=[100, 50, 30],
        )
        assert "data" in result
        assert result["data"][0]["type"] == "bar"

    def test_model_comparison_chart(self):
        result = ChartGenerator.model_comparison_chart(
            model_names=["model_a", "model_b"],
            metrics={
                "accuracy": [0.95, 0.92],
                "f1_score": [0.93, 0.90],
            },
        )
        assert "data" in result
        assert len(result["data"]) == 2

    def test_confusion_matrix_chart(self):
        result = ChartGenerator.confusion_matrix_chart(
            labels=["cat", "dog"],
            matrix=[[45, 5], [3, 47]],
        )
        assert "data" in result
        assert result["data"][0]["type"] == "heatmap"

    def test_chart_has_layout(self):
        result = ChartGenerator.latency_timeseries(
            timestamps=["t1"], p50=[10.0], p95=[50.0], p99=[100.0]
        )
        assert "layout" in result
        assert "title" in result["layout"]

    def test_empty_data_no_error(self):
        result = ChartGenerator.throughput_chart(timestamps=[], rps=[])
        assert "data" in result

    def test_single_point(self):
        result = ChartGenerator.error_rate_chart(
            timestamps=["2024-01-01"],
            error_rates=[0.5],
        )
        assert "data" in result
