"""Async database layer with SQLAlchemy + aiosqlite."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    String,
    Text,
    func,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from src.config.settings import settings


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_name = Column(String, nullable=False, index=True)
    model_version = Column(String, nullable=False)
    input_hash = Column(String, nullable=False)
    prediction = Column(Text, nullable=False)
    actual = Column(Text, nullable=True)
    latency_ms = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class MetricsSnapshot(Base):
    __tablename__ = "metrics_snapshots"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_name = Column(String, nullable=False, index=True)
    metric_name = Column(String, nullable=False, index=True)
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class AlertRecord(Base):
    __tablename__ = "alerts"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    rule_name = Column(String, nullable=False)
    severity = Column(String, nullable=False, default="info")
    message = Column(Text, nullable=False)
    acknowledged = Column(Boolean, nullable=False, default=False)
    timestamp = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class ExperimentRecord(Base):
    __tablename__ = "experiments"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, unique=True)
    model_a = Column(String, nullable=False)
    model_b = Column(String, nullable=False)
    status = Column(String, nullable=False, default="running")
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class ModelRecord(Base):
    __tablename__ = "models"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, index=True)
    version = Column(String, nullable=False)
    stage = Column(String, nullable=False, default="staging")
    registered_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class Database:
    """Async database manager for all ML monitoring data."""

    def __init__(self, db_url: str | None = None) -> None:
        self.db_url = db_url or settings.DB_URL
        self.engine = create_async_engine(self.db_url, echo=False)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def initialize(self) -> None:
        """Create all tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        """Close the engine."""
        await self.engine.dispose()

    def session(self) -> AsyncSession:
        """Get a new async session."""
        return self.session_factory()

    # --- Predictions ---

    async def insert_predictions(self, records: list[dict[str, Any]]) -> list[str]:
        """Batch insert prediction records. Returns list of IDs."""
        ids = []
        async with self.session() as session:
            async with session.begin():
                for rec in records:
                    pred = PredictionRecord(**rec)
                    if not pred.id:
                        pred.id = str(uuid.uuid4())
                    session.add(pred)
                    ids.append(pred.id)
        return ids

    async def get_predictions(
        self,
        model_name: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> list[PredictionRecord]:
        """Query predictions with optional time range filter."""
        async with self.session() as session:
            stmt = select(PredictionRecord).where(PredictionRecord.model_name == model_name)
            if start_time:
                stmt = stmt.where(PredictionRecord.timestamp >= start_time)
            if end_time:
                stmt = stmt.where(PredictionRecord.timestamp <= end_time)
            stmt = stmt.order_by(PredictionRecord.timestamp.desc()).limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def update_ground_truth(self, prediction_id: str, actual_value: str) -> bool:
        """Update a prediction with the actual ground truth value."""
        async with self.session() as session:
            async with session.begin():
                stmt = select(PredictionRecord).where(PredictionRecord.id == prediction_id)
                result = await session.execute(stmt)
                pred = result.scalar_one_or_none()
                if pred is None:
                    return False
                pred.actual = actual_value
        return True

    # --- Metrics ---

    async def insert_metric(
        self, model_name: str, metric_name: str, value: float
    ) -> str:
        """Insert a single metric snapshot."""
        rec = MetricsSnapshot(
            id=str(uuid.uuid4()),
            model_name=model_name,
            metric_name=metric_name,
            value=value,
        )
        async with self.session() as session:
            async with session.begin():
                session.add(rec)
        return rec.id

    async def get_metrics(
        self,
        model_name: str,
        metric_name: str | None = None,
        start_time: datetime | None = None,
        limit: int = 500,
    ) -> list[MetricsSnapshot]:
        """Query metric snapshots."""
        async with self.session() as session:
            stmt = select(MetricsSnapshot).where(MetricsSnapshot.model_name == model_name)
            if metric_name:
                stmt = stmt.where(MetricsSnapshot.metric_name == metric_name)
            if start_time:
                stmt = stmt.where(MetricsSnapshot.timestamp >= start_time)
            stmt = stmt.order_by(MetricsSnapshot.timestamp.desc()).limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_metric_aggregates(
        self, model_name: str, metric_name: str, start_time: datetime
    ) -> dict[str, float | None]:
        """Get aggregate stats (avg, min, max, count) for a metric."""
        async with self.session() as session:
            stmt = (
                select(
                    func.avg(MetricsSnapshot.value).label("avg"),
                    func.min(MetricsSnapshot.value).label("min"),
                    func.max(MetricsSnapshot.value).label("max"),
                    func.count(MetricsSnapshot.id).label("count"),
                )
                .where(MetricsSnapshot.model_name == model_name)
                .where(MetricsSnapshot.metric_name == metric_name)
                .where(MetricsSnapshot.timestamp >= start_time)
            )
            result = await session.execute(stmt)
            row = result.one()
            return {"avg": row.avg, "min": row.min, "max": row.max, "count": row.count}

    # --- Alerts ---

    async def insert_alert(self, alert_data: dict[str, Any]) -> str:
        """Insert an alert record."""
        alert = AlertRecord(**alert_data)
        if not alert.id:
            alert.id = str(uuid.uuid4())
        async with self.session() as session:
            async with session.begin():
                session.add(alert)
        return alert.id

    async def get_active_alerts(self) -> list[AlertRecord]:
        """Get all unacknowledged alerts."""
        async with self.session() as session:
            stmt = (
                select(AlertRecord)
                .where(AlertRecord.acknowledged == False)  # noqa: E712
                .order_by(AlertRecord.timestamp.desc())
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged."""
        async with self.session() as session:
            async with session.begin():
                stmt = select(AlertRecord).where(AlertRecord.id == alert_id)
                result = await session.execute(stmt)
                alert = result.scalar_one_or_none()
                if alert is None:
                    return False
                alert.acknowledged = True
        return True

    # --- Experiments ---

    async def insert_experiment(self, experiment_data: dict[str, Any]) -> str:
        """Create an experiment."""
        exp = ExperimentRecord(**experiment_data)
        if not exp.id:
            exp.id = str(uuid.uuid4())
        async with self.session() as session:
            async with session.begin():
                session.add(exp)
        return exp.id

    async def get_experiments(self) -> list[ExperimentRecord]:
        """List all experiments."""
        async with self.session() as session:
            stmt = select(ExperimentRecord).order_by(ExperimentRecord.created_at.desc())
            result = await session.execute(stmt)
            return list(result.scalars().all())

    # --- Models ---

    async def register_model(self, model_data: dict[str, Any]) -> str:
        """Register a model."""
        model = ModelRecord(**model_data)
        if not model.id:
            model.id = str(uuid.uuid4())
        async with self.session() as session:
            async with session.begin():
                session.add(model)
        return model.id

    async def get_models(self) -> list[ModelRecord]:
        """List all registered models."""
        async with self.session() as session:
            stmt = select(ModelRecord).order_by(ModelRecord.registered_at.desc())
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_model_by_name(self, name: str) -> ModelRecord | None:
        """Get a model by name."""
        async with self.session() as session:
            stmt = select(ModelRecord).where(ModelRecord.name == name).limit(1)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()
