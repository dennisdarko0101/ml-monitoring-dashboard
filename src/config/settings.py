"""Application settings with environment variable support."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for the ML monitoring platform."""

    model_config = {"env_prefix": "ML_MONITOR_", "env_file": ".env", "extra": "ignore"}

    # Database
    DB_URL: str = "sqlite+aiosqlite:///./ml_monitor.db"

    # Alerting
    ALERT_WEBHOOK: str = ""
    ALERT_EMAIL_RECIPIENTS: str = ""

    # Monitoring intervals
    CHECK_INTERVAL_SECONDS: int = 60
    FLUSH_INTERVAL_SECONDS: float = 5.0
    FLUSH_BATCH_SIZE: int = 100

    # Drift thresholds
    DRIFT_THRESHOLD: float = 0.05
    PSI_THRESHOLD: float = 0.2
    JS_DIVERGENCE_THRESHOLD: float = 0.1

    # Performance thresholds
    LATENCY_P99_THRESHOLD: float = 500.0
    ERROR_RATE_THRESHOLD: float = 0.05
    THROUGHPUT_MIN_THRESHOLD: float = 1.0

    # Dashboard
    DASHBOARD_TITLE: str = "ML Monitoring Dashboard"
    DEFAULT_WINDOW_HOURS: int = 24
    # When true, seed a synthetic demo model on startup and serve the demo drift
    # report so the dashboard renders with content out of the box.
    DEMO_MODE: bool = True

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"


settings = Settings()
