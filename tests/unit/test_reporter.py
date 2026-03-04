"""Tests for ReportGenerator: daily, weekly, export formats."""

from __future__ import annotations

import pytest

from src.collectors.metrics_collector import MetricsCollector
from src.dashboard.charts import ChartGenerator
from src.dashboard.reporter import Report, ReportGenerator
from src.detectors.data_drift import DriftReport, FeatureDriftResult


@pytest.fixture
def reporter(metrics_collector):
    return ReportGenerator(metrics_collector, ChartGenerator())


@pytest.fixture
def sample_drift_report():
    return DriftReport(
        per_feature_scores=[
            FeatureDriftResult(
                feature="feat_a", drift_score=0.8, method="ks_test",
                p_value=0.001, is_drifted=True,
            ),
        ],
        overall_drift_score=0.8,
        drifted_features=["feat_a"],
        is_drifted=True,
    )


class TestReportGenerator:
    def test_daily_report(self, reporter, metrics_collector):
        metrics_collector.record_prediction("model_a", 50.0)
        report = reporter.daily_report("model_a")
        assert report.model_name == "model_a"
        assert report.period == "daily"
        assert "Latency P50" in report.summary_stats

    def test_weekly_report(self, reporter, metrics_collector):
        for i in range(10):
            metrics_collector.record_prediction("model_a", float(i * 10))
        report = reporter.weekly_report("model_a")
        assert report.period == "weekly"

    def test_report_with_drift(self, reporter, metrics_collector, sample_drift_report):
        metrics_collector.record_prediction("m", 50.0)
        report = reporter.daily_report("m", drift_report=sample_drift_report)
        assert report.drift_status["is_drifted"] is True
        assert "feat_a" in report.drift_status["drifted_features"]

    def test_report_with_alerts(self, reporter, metrics_collector):
        alerts = [{"severity": "critical", "message": "High error rate"}]
        report = reporter.daily_report("m", alerts=alerts)
        assert len(report.alerts) == 1

    def test_recommendations_high_latency(self, reporter, metrics_collector):
        for _ in range(100):
            metrics_collector.record_prediction("m", 600.0)
        report = reporter.daily_report("m")
        assert any("latency" in r.lower() for r in report.recommendations)

    def test_recommendations_high_error_rate(self, reporter, metrics_collector):
        for _ in range(100):
            metrics_collector.record_prediction("m", 10.0, is_error=True)
        report = reporter.daily_report("m")
        assert any("error" in r.lower() for r in report.recommendations)

    def test_recommendations_drift(self, reporter, metrics_collector, sample_drift_report):
        report = reporter.daily_report("m", drift_report=sample_drift_report)
        assert any("drift" in r.lower() for r in report.recommendations)

    def test_no_issues_recommendation(self, reporter):
        report = reporter.daily_report("m")
        assert any("no action" in r.lower() for r in report.recommendations)


class TestReportExport:
    def test_to_json(self, reporter):
        report = reporter.daily_report("m")
        json_str = report.to_json()
        assert '"model_name"' in json_str
        assert '"m"' in json_str

    def test_to_markdown(self, reporter, metrics_collector):
        metrics_collector.record_prediction("m", 50.0)
        report = reporter.daily_report("m")
        md = report.to_markdown()
        assert "# Monitoring Report" in md
        assert "## Summary" in md

    def test_to_html(self, reporter, metrics_collector):
        metrics_collector.record_prediction("m", 50.0)
        report = reporter.daily_report("m")
        html = report.to_html()
        assert "<html>" in html
        assert "Monitoring Report" in html
