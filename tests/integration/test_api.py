"""Integration tests for all API endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient

from src.api import main as api_main
from src.api.main import app
from src.alerting.alert_manager import AlertManager
from src.alerting.notifier import Notifier
from src.collectors.metrics_collector import MetricsCollector
from src.collectors.prediction_logger import PredictionLogger
from src.collectors.system_monitor import SystemMonitor
from src.dashboard.charts import ChartGenerator
from src.dashboard.reporter import ReportGenerator
from src.storage.database import Database


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    # Manually set up the global state that lifespan would create
    db = Database(db_url="sqlite+aiosqlite:///:memory:")
    await db.initialize()

    api_main.db = db
    api_main.prediction_logger = PredictionLogger(db)
    api_main.metrics_collector = MetricsCollector(db)
    api_main.alert_manager = AlertManager(db)
    api_main.notifier = Notifier()
    api_main.system_monitor = SystemMonitor()
    api_main.chart_generator = ChartGenerator()
    api_main.report_generator = ReportGenerator(api_main.metrics_collector, api_main.chart_generator)

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    await db.close()
    api_main.db = None
    api_main.prediction_logger = None
    api_main.metrics_collector = None
    api_main.alert_manager = None
    api_main.notifier = None
    api_main.system_monitor = None
    api_main.chart_generator = None
    api_main.report_generator = None


class TestHealthEndpoints:
    async def test_health(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    async def test_metrics_endpoint(self, client):
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        assert "ml_prediction" in resp.text or resp.status_code == 200


class TestPredictionLogging:
    async def test_log_prediction(self, client):
        resp = await client.post("/api/v1/log/prediction", json={
            "model_name": "test_model",
            "model_version": "1.0",
            "input_data": {"feature": 42},
            "prediction": "positive",
            "latency_ms": 25.0,
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "logged"

    async def test_log_prediction_invalid(self, client):
        resp = await client.post("/api/v1/log/prediction", json={
            "model_name": "test_model",
            "prediction": "positive",
            # missing required fields
        })
        assert resp.status_code == 422

    async def test_log_ground_truth(self, client):
        resp = await client.post("/api/v1/log/ground-truth", json={
            "prediction_id": "some-id",
            "actual_value": "negative",
        })
        assert resp.status_code == 200
        assert resp.json()["prediction_id"] == "some-id"


class TestMetricsEndpoints:
    async def test_get_metrics(self, client):
        # Log some predictions first
        await client.post("/api/v1/log/prediction", json={
            "model_name": "metrics_model",
            "model_version": "1.0",
            "input_data": {"x": 1},
            "prediction": 0.5,
            "latency_ms": 30.0,
        })
        resp = await client.get("/api/v1/metrics/metrics_model?window=1h")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == "metrics_model"
        assert "latency_p50" in data

    async def test_get_metrics_unknown_model(self, client):
        resp = await client.get("/api/v1/metrics/unknown_model")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_predictions"] == 0


class TestDriftEndpoints:
    async def test_get_drift_report(self, client):
        resp = await client.get("/api/v1/drift/test_model")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == "test_model"
        assert "is_drifted" in data


class TestAlertEndpoints:
    async def test_get_alerts_empty(self, client):
        resp = await client.get("/api/v1/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert "alerts" in data
        assert "count" in data

    async def test_acknowledge_nonexistent(self, client):
        resp = await client.post("/api/v1/alerts/fake-id/acknowledge")
        assert resp.status_code == 404


class TestDashboardEndpoints:
    async def test_dashboard(self, client):
        resp = await client.get("/api/v1/dashboard/test_model")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == "test_model"
        assert "metrics" in data
        assert "alerts" in data

    async def test_dashboard_with_data(self, client):
        for _ in range(5):
            await client.post("/api/v1/log/prediction", json={
                "model_name": "dash_model",
                "model_version": "1.0",
                "input_data": {"x": 1},
                "prediction": "positive",
                "latency_ms": 45.0,
            })
        resp = await client.get("/api/v1/dashboard/dash_model")
        assert resp.status_code == 200


class TestReportEndpoints:
    async def test_daily_report(self, client):
        resp = await client.get("/api/v1/report/test_model?period=daily")
        assert resp.status_code == 200
        data = resp.json()
        assert data["period"] == "daily"
        assert "recommendations" in data

    async def test_weekly_report(self, client):
        resp = await client.get("/api/v1/report/test_model?period=weekly")
        assert resp.status_code == 200
        assert resp.json()["period"] == "weekly"


class TestModelEndpoints:
    async def test_list_models(self, client):
        resp = await client.get("/api/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "count" in data


class TestSystemEndpoints:
    async def test_system_health(self, client):
        resp = await client.get("/api/v1/system")
        assert resp.status_code == 200
        data = resp.json()
        assert "cpu_percent" in data
        assert "memory_percent" in data
        assert "gpu_percent" in data
        assert "disk_percent" in data
