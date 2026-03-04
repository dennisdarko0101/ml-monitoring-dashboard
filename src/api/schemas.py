"""Pydantic models for API request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# --- Requests ---


class PredictionLogRequest(BaseModel):
    model_name: str
    model_version: str = "1.0.0"
    input_data: Any
    prediction: Any
    latency_ms: float = Field(ge=0)


class GroundTruthRequest(BaseModel):
    prediction_id: str
    actual_value: Any


class AcknowledgeAlertRequest(BaseModel):
    pass


# --- Responses ---


class PredictionLogResponse(BaseModel):
    status: str = "logged"
    buffer_size: int = 0


class GroundTruthResponse(BaseModel):
    status: str = "updated"
    prediction_id: str


class MetricsSummaryResponse(BaseModel):
    model_name: str
    window_hours: int
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput_rps: float
    error_rate: float
    total_predictions: int
    total_errors: int
    prediction_distribution: dict[str, int]


class FeatureDriftResponse(BaseModel):
    feature: str
    drift_score: float
    method: str
    p_value: float | None = None
    is_drifted: bool


class DriftReportResponse(BaseModel):
    model_name: str
    overall_drift_score: float
    is_drifted: bool
    drifted_features: list[str]
    per_feature_scores: list[FeatureDriftResponse]


class AlertResponse(BaseModel):
    id: str
    rule_name: str
    severity: str
    message: str
    timestamp: float
    acknowledged: bool


class AlertListResponse(BaseModel):
    alerts: list[AlertResponse]
    count: int


class DashboardResponse(BaseModel):
    model_name: str
    metrics: MetricsSummaryResponse
    charts: dict[str, Any]
    alerts: list[AlertResponse]
    drift: DriftReportResponse | None = None


class ReportResponse(BaseModel):
    model_name: str
    period: str
    generated_at: str
    summary_stats: dict[str, Any]
    alerts: list[dict[str, Any]]
    drift_status: dict[str, Any]
    recommendations: list[str]


class ModelResponse(BaseModel):
    id: str
    name: str
    version: str
    stage: str
    registered_at: str


class ModelListResponse(BaseModel):
    models: list[ModelResponse]
    count: int


class SystemHealthResponse(BaseModel):
    cpu_percent: float
    memory_percent: float
    gpu_percent: float
    disk_percent: float
    timestamp: float


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
