"""Integration tests for the full monitoring pipeline: log → detect → alert flow."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.alerting.alert_manager import AlertManager, AlertSeverity
from src.alerting.notifier import Notifier
from src.collectors.metrics_collector import MetricsCollector
from src.collectors.prediction_logger import PredictionLogger
from src.dashboard.charts import ChartGenerator
from src.dashboard.reporter import ReportGenerator
from src.detectors.anomaly_detector import AnomalyDetector
from src.detectors.data_drift import DataDriftDetector
from src.detectors.model_drift import ModelDriftDetector
from src.storage.database import Database


class TestPipelineLogToDetect:
    """Test the flow: log predictions → collect metrics → detect drift."""

    async def test_log_and_collect_metrics(self, db):
        logger = PredictionLogger(db)
        collector = MetricsCollector(db)

        for i in range(50):
            latency = 20.0 + i * 0.5
            await logger.log("pipeline_model", "1.0", {"x": i}, float(i % 2), latency)
            collector.record_prediction("pipeline_model", latency)

        await logger.flush()
        summary = collector.get_summary("pipeline_model", "1h")
        assert summary.total_predictions == 50
        assert summary.latency_p50 > 0

    async def test_collect_then_detect_data_drift(self, db):
        rng = np.random.RandomState(42)
        ref_data = pd.DataFrame({"feature": rng.normal(0, 1, 500)})
        cur_data = pd.DataFrame({"feature": rng.normal(3, 1, 500)})

        detector = DataDriftDetector()
        detector.set_reference(ref_data)
        report = detector.detect(cur_data)
        assert report.is_drifted

    async def test_collect_then_detect_model_drift(self, db):
        detector = ModelDriftDetector(ph_threshold=5.0)
        predictions = np.array([1] * 200)
        actuals = np.array([1] * 100 + [0] * 100)  # concept drift
        report = detector.detect(predictions, actuals)
        assert report.concept_drift_detected


class TestPipelineDetectToAlert:
    """Test the flow: detect → alert → notify."""

    async def test_drift_triggers_alert(self, db):
        manager = AlertManager(db)
        drift_score = 0.8  # above threshold of 0.5
        metrics = {"drift_score": drift_score}
        alerts = manager.check_all(metrics)
        drift_alerts = [a for a in alerts if a.rule_name == "drift_detected"]
        assert len(drift_alerts) == 1

    async def test_high_latency_triggers_alert(self, db):
        manager = AlertManager(db)
        metrics = {"latency_p99": 600.0}
        alerts = manager.check_all(metrics)
        assert any(a.rule_name == "high_latency" for a in alerts)

    async def test_alert_notification_flow(self, db):
        manager = AlertManager(db)
        notifier = Notifier()

        metrics = {"error_rate": 0.15}
        alerts = manager.check_all(metrics)
        assert len(alerts) > 0

        for alert in alerts:
            results = await notifier.notify(alert)
            assert results["log"] is True


class TestPipelineAlertToVisualize:
    """Test the flow: alert → visualize → report."""

    async def test_full_report_pipeline(self, db):
        collector = MetricsCollector(db)
        for i in range(100):
            collector.record_prediction("full_model", float(i), "label_a")

        manager = AlertManager(db)
        metrics = {"latency_p99": 600.0, "error_rate": 0.1}
        fired = manager.check_all(metrics)

        reporter = ReportGenerator(collector)
        alert_dicts = [
            {"severity": a.severity.value, "message": a.message} for a in fired
        ]
        report = reporter.daily_report("full_model", alerts=alert_dicts)
        assert len(report.alerts) > 0
        assert len(report.recommendations) > 0

    async def test_chart_from_metrics(self, db):
        collector = MetricsCollector(db)
        for i in range(20):
            collector.record_prediction("chart_model", 10.0 + i, "positive")

        summary = collector.get_summary("chart_model", "1h")
        dist = summary.prediction_distribution
        chart = ChartGenerator.prediction_distribution(
            list(dist.keys()), list(dist.values())
        )
        assert "data" in chart


class TestEndToEndPipeline:
    """Full end-to-end: log → metrics → drift → anomaly → alert → report."""

    async def test_complete_pipeline(self, db):
        # 1. Log predictions
        logger = PredictionLogger(db)
        collector = MetricsCollector(db)
        rng = np.random.RandomState(42)

        latencies = []
        for i in range(200):
            latency = float(rng.exponential(50))
            await logger.log("e2e_model", "2.0", {"x": i}, i % 3, latency)
            collector.record_prediction("e2e_model", latency, str(i % 3))
            latencies.append(latency)
        await logger.flush()

        # 2. Detect anomalies
        detector = AnomalyDetector()
        anomalies = detector.detect_latency_anomalies(latencies)
        # Some exponential samples might be anomalous

        # 3. Data drift detection
        ref = pd.DataFrame({"feat": rng.normal(0, 1, 200)})
        cur = pd.DataFrame({"feat": rng.normal(2, 1, 200)})
        drift_detector = DataDriftDetector()
        drift_detector.set_reference(ref)
        drift_report = drift_detector.detect(cur)

        # 4. Alert check
        manager = AlertManager(db)
        summary = collector.get_summary("e2e_model", "1h")
        metrics_dict = {
            "latency_p99": summary.latency_p99,
            "error_rate": summary.error_rate,
            "drift_score": drift_report.overall_drift_score,
        }
        alerts = manager.check_all(metrics_dict)

        # 5. Generate report
        reporter = ReportGenerator(collector)
        alert_dicts = [
            {"severity": a.severity.value, "message": a.message} for a in alerts
        ]
        report = reporter.daily_report(
            "e2e_model", drift_report=drift_report, alerts=alert_dicts
        )

        assert report.model_name == "e2e_model"
        assert report.period == "daily"
        assert "Latency P50" in report.summary_stats
        assert isinstance(report.to_json(), str)
        assert isinstance(report.to_markdown(), str)

    async def test_database_persistence(self, db):
        """Verify data persists through the pipeline."""
        logger = PredictionLogger(db)
        for i in range(10):
            await logger.log("persist_model", "1.0", {"i": i}, i, 10.0 + i)
        await logger.flush()

        preds = await db.get_predictions("persist_model")
        assert len(preds) == 10

        # Insert and query metrics
        await db.insert_metric("persist_model", "latency_p99", 45.0)
        metrics = await db.get_metrics("persist_model", "latency_p99")
        assert len(metrics) == 1

        # Insert and query alerts
        alert_id = await db.insert_alert({
            "rule_name": "test_rule",
            "severity": "warning",
            "message": "Test alert",
        })
        active = await db.get_active_alerts()
        assert any(a.id == alert_id for a in active)

        # Acknowledge
        await db.acknowledge_alert(alert_id)
        active = await db.get_active_alerts()
        assert all(a.id != alert_id for a in active)
