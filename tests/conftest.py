"""Shared fixtures for all tests."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio

from src.alerting.alert_manager import AlertManager
from src.collectors.metrics_collector import MetricsCollector
from src.collectors.prediction_logger import PredictionLogger
from src.detectors.anomaly_detector import AnomalyDetector
from src.detectors.data_drift import DataDriftDetector
from src.detectors.model_drift import ModelDriftDetector
from src.storage.database import Database


@pytest_asyncio.fixture
async def db():
    """In-memory SQLite database for tests."""
    database = Database(db_url="sqlite+aiosqlite:///:memory:")
    await database.initialize()
    yield database
    await database.close()


@pytest_asyncio.fixture
async def prediction_logger(db):
    """PredictionLogger backed by in-memory DB."""
    logger = PredictionLogger(db)
    yield logger


@pytest.fixture
def metrics_collector(db):
    """MetricsCollector backed by in-memory DB."""
    return MetricsCollector(db)


@pytest.fixture
def alert_manager(db):
    """AlertManager with default rules."""
    return AlertManager(db)


@pytest.fixture
def data_drift_detector():
    """DataDriftDetector with default thresholds."""
    return DataDriftDetector()


@pytest.fixture
def model_drift_detector():
    """ModelDriftDetector with default settings."""
    return ModelDriftDetector()


@pytest.fixture
def anomaly_detector():
    """AnomalyDetector with default settings."""
    return AnomalyDetector()


# --- Synthetic data generators ---


@pytest.fixture
def normal_numeric_df():
    """Generate a reference DataFrame with normally distributed numeric features."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "feature_a": rng.normal(0, 1, 500),
        "feature_b": rng.normal(10, 2, 500),
        "feature_c": rng.normal(-5, 0.5, 500),
    })


@pytest.fixture
def shifted_numeric_df():
    """Generate a current DataFrame with shifted distributions (drift)."""
    rng = np.random.RandomState(99)
    return pd.DataFrame({
        "feature_a": rng.normal(3, 1, 500),      # shifted mean
        "feature_b": rng.normal(10, 5, 500),      # widened std
        "feature_c": rng.normal(-5, 0.5, 500),    # no drift
    })


@pytest.fixture
def categorical_ref_df():
    """Reference categorical data."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "color": rng.choice(["red", "blue", "green"], 500, p=[0.5, 0.3, 0.2]),
        "size": rng.choice(["S", "M", "L", "XL"], 500, p=[0.1, 0.4, 0.35, 0.15]),
    })


@pytest.fixture
def categorical_shifted_df(categorical_ref_df):
    """Shifted categorical data (drift on color only, size is copied from ref)."""
    rng = np.random.RandomState(99)
    return pd.DataFrame({
        "color": rng.choice(["red", "blue", "green"], 500, p=[0.1, 0.1, 0.8]),  # very different
        "size": categorical_ref_df["size"].values.copy(),  # exact same data -> no drift
    })


@pytest.fixture
def normal_latencies():
    """Normal latency data with a few outliers."""
    rng = np.random.RandomState(42)
    base = rng.normal(50, 10, 200).tolist()
    base[100] = 500.0   # outlier
    base[150] = 600.0   # outlier
    return base


@pytest.fixture
def normal_error_counts():
    """Normal error counts with a spike."""
    rng = np.random.RandomState(42)
    base = rng.poisson(2, 200).astype(float).tolist()
    base[120] = 50.0    # spike
    base[121] = 45.0    # spike
    return base
