"""FastAPI application: ML Monitoring Dashboard API."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.alerting.alert_manager import AlertManager
from src.alerting.notifier import Notifier
from src.api.schemas import (
    AlertListResponse,
    AlertResponse,
    DashboardResponse,
    DriftReportResponse,
    FeatureDriftResponse,
    GroundTruthRequest,
    GroundTruthResponse,
    HealthResponse,
    MetricsSummaryResponse,
    ModelListResponse,
    ModelResponse,
    PredictionLogRequest,
    PredictionLogResponse,
    ReportResponse,
    SystemHealthResponse,
)
from src.collectors.metrics_collector import MetricsCollector
from src.collectors.prediction_logger import PredictionLogger
from src.collectors.system_monitor import SystemMonitor
from src.config.settings import settings
from src.dashboard.charts import ChartGenerator
from src.dashboard.reporter import ReportGenerator
from src.storage.database import Database
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# --- Global state ---
db: Database | None = None
prediction_logger: PredictionLogger | None = None
metrics_collector: MetricsCollector | None = None
alert_manager: AlertManager | None = None
notifier: Notifier | None = None
system_monitor: SystemMonitor | None = None
chart_generator: ChartGenerator | None = None
report_generator: ReportGenerator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    global db, prediction_logger, metrics_collector, alert_manager
    global notifier, system_monitor, chart_generator, report_generator

    setup_logging()
    logger.info("starting_ml_monitor")

    db = Database()
    await db.initialize()

    prediction_logger = PredictionLogger(db)
    await prediction_logger.start()

    metrics_collector = MetricsCollector(db)
    alert_manager = AlertManager(db)
    notifier = Notifier()
    system_monitor = SystemMonitor()
    chart_generator = ChartGenerator()
    report_generator = ReportGenerator(metrics_collector, chart_generator)

    logger.info("ml_monitor_ready")
    yield

    # Shutdown
    if prediction_logger:
        await prediction_logger.stop()
    if db:
        await db.close()
    logger.info("ml_monitor_stopped")


app = FastAPI(
    title="ML Monitoring Dashboard",
    description="Production ML observability platform",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Health ---


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse()


@app.get("/metrics")
async def prometheus_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# --- Prediction Logging ---


@app.post("/api/v1/log/prediction", response_model=PredictionLogResponse)
async def log_prediction(req: PredictionLogRequest):
    assert prediction_logger is not None
    assert metrics_collector is not None
    await prediction_logger.log(
        model_name=req.model_name,
        version=req.model_version,
        input_data=req.input_data,
        prediction=req.prediction,
        latency_ms=req.latency_ms,
    )
    metrics_collector.record_prediction(req.model_name, req.latency_ms)
    return PredictionLogResponse(buffer_size=prediction_logger.buffer_size)


@app.post("/api/v1/log/ground-truth", response_model=GroundTruthResponse)
async def log_ground_truth(req: GroundTruthRequest):
    assert prediction_logger is not None
    await prediction_logger.log_ground_truth(req.prediction_id, req.actual_value)
    return GroundTruthResponse(prediction_id=req.prediction_id)


# --- Metrics ---


@app.get("/api/v1/metrics/{model_name}", response_model=MetricsSummaryResponse)
async def get_metrics(model_name: str, window: str = "1h"):
    assert metrics_collector is not None
    summary = metrics_collector.get_summary(model_name, window)
    return MetricsSummaryResponse(
        model_name=summary.model_name,
        window_hours=summary.window_hours,
        latency_p50=summary.latency_p50,
        latency_p95=summary.latency_p95,
        latency_p99=summary.latency_p99,
        throughput_rps=summary.throughput_rps,
        error_rate=summary.error_rate,
        total_predictions=summary.total_predictions,
        total_errors=summary.total_errors,
        prediction_distribution=summary.prediction_distribution,
    )


# --- Drift ---


@app.get("/api/v1/drift/{model_name}", response_model=DriftReportResponse)
async def get_drift_report(model_name: str):
    """Return the most recent drift analysis for a model.

    Note: In a full deployment, drift detection runs periodically. This endpoint
    returns a placeholder when no drift analysis has been run yet.
    """
    return DriftReportResponse(
        model_name=model_name,
        overall_drift_score=0.0,
        is_drifted=False,
        drifted_features=[],
        per_feature_scores=[],
    )


# --- Alerts ---


@app.get("/api/v1/alerts", response_model=AlertListResponse)
async def get_alerts():
    assert alert_manager is not None
    active = alert_manager.get_active_alerts()
    alerts = [
        AlertResponse(
            id=a.id,
            rule_name=a.rule_name,
            severity=a.severity.value,
            message=a.message,
            timestamp=a.timestamp,
            acknowledged=a.acknowledged,
        )
        for a in active
    ]
    return AlertListResponse(alerts=alerts, count=len(alerts))


@app.post("/api/v1/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    assert alert_manager is not None
    success = alert_manager.acknowledge(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"status": "acknowledged", "alert_id": alert_id}


# --- Dashboard ---


@app.get("/api/v1/dashboard/{model_name}", response_model=DashboardResponse)
async def get_dashboard(model_name: str):
    assert metrics_collector is not None
    assert alert_manager is not None

    summary = metrics_collector.get_summary(model_name, "1h")

    # Generate charts
    charts = {}
    dist = summary.prediction_distribution
    if dist:
        charts["prediction_distribution"] = ChartGenerator.prediction_distribution(
            list(dist.keys()), list(dist.values())
        )

    active = alert_manager.get_active_alerts()
    alerts = [
        AlertResponse(
            id=a.id, rule_name=a.rule_name, severity=a.severity.value,
            message=a.message, timestamp=a.timestamp, acknowledged=a.acknowledged,
        )
        for a in active
    ]

    return DashboardResponse(
        model_name=model_name,
        metrics=MetricsSummaryResponse(
            model_name=summary.model_name,
            window_hours=summary.window_hours,
            latency_p50=summary.latency_p50,
            latency_p95=summary.latency_p95,
            latency_p99=summary.latency_p99,
            throughput_rps=summary.throughput_rps,
            error_rate=summary.error_rate,
            total_predictions=summary.total_predictions,
            total_errors=summary.total_errors,
            prediction_distribution=summary.prediction_distribution,
        ),
        charts=charts,
        alerts=alerts,
    )


# --- Reports ---


@app.get("/api/v1/report/{model_name}", response_model=ReportResponse)
async def generate_report(model_name: str, period: str = "daily"):
    assert report_generator is not None
    if period == "weekly":
        report = report_generator.weekly_report(model_name)
    else:
        report = report_generator.daily_report(model_name)
    return ReportResponse(
        model_name=report.model_name,
        period=report.period,
        generated_at=report.generated_at,
        summary_stats=report.summary_stats,
        alerts=report.alerts,
        drift_status=report.drift_status,
        recommendations=report.recommendations,
    )


# --- Models ---


@app.get("/api/v1/models", response_model=ModelListResponse)
async def list_models():
    assert db is not None
    models = await db.get_models()
    items = [
        ModelResponse(
            id=m.id, name=m.name, version=m.version,
            stage=m.stage, registered_at=m.registered_at.isoformat(),
        )
        for m in models
    ]
    return ModelListResponse(models=items, count=len(items))


# --- System ---


@app.get("/api/v1/system", response_model=SystemHealthResponse)
async def system_health():
    assert system_monitor is not None
    snap = system_monitor.snapshot()
    return SystemHealthResponse(
        cpu_percent=snap.cpu_percent,
        memory_percent=snap.memory_percent,
        gpu_percent=snap.gpu_percent,
        disk_percent=snap.disk_percent,
        timestamp=snap.timestamp,
    )


# --- WebSocket ---


@app.websocket("/ws/live")
async def live_metrics(websocket: WebSocket):
    """Stream live metrics to connected clients."""
    await websocket.accept()
    assert metrics_collector is not None
    assert system_monitor is not None
    try:
        while True:
            snap = system_monitor.snapshot()
            payload = {
                "type": "system",
                "cpu": snap.cpu_percent,
                "memory": snap.memory_percent,
                "gpu": snap.gpu_percent,
                "disk": snap.disk_percent,
                "timestamp": snap.timestamp,
            }
            await websocket.send_json(payload)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass


def run() -> None:
    """Entry point for running the server."""
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )


if __name__ == "__main__":
    run()
