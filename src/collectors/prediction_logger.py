"""High-throughput prediction logger with buffered writes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any

from src.config.settings import settings
from src.storage.database import Database
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PredictionLogger:
    """Buffers prediction logs and flushes them in batches for throughput."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self._buffer: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task[None] | None = None
        self._running = False
        self.batch_size = settings.FLUSH_BATCH_SIZE
        self.flush_interval = settings.FLUSH_INTERVAL_SECONDS

    async def start(self) -> None:
        """Start the background flush loop."""
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        """Stop the flush loop and drain remaining buffer."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self.flush()

    async def log(
        self,
        model_name: str,
        version: str,
        input_data: Any,
        prediction: Any,
        latency_ms: float,
    ) -> None:
        """Buffer a prediction record for async batch insert."""
        input_hash = hashlib.sha256(
            json.dumps(input_data, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        record = {
            "model_name": model_name,
            "model_version": version,
            "input_hash": input_hash,
            "prediction": json.dumps(prediction, default=str),
            "latency_ms": latency_ms,
            "timestamp": datetime.now(timezone.utc),
        }

        async with self._lock:
            self._buffer.append(record)
            if len(self._buffer) >= self.batch_size:
                await self._do_flush()

    async def log_ground_truth(self, prediction_id: str, actual_value: Any) -> None:
        """Attach a delayed ground-truth label to an existing prediction."""
        actual_str = json.dumps(actual_value, default=str)
        updated = await self.db.update_ground_truth(prediction_id, actual_str)
        if not updated:
            logger.warning("prediction_not_found", prediction_id=prediction_id)

    async def get_predictions(
        self,
        model_name: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve logged predictions for a model within a time range."""
        records = await self.db.get_predictions(model_name, start_time, end_time)
        return [
            {
                "id": r.id,
                "model_name": r.model_name,
                "model_version": r.model_version,
                "input_hash": r.input_hash,
                "prediction": r.prediction,
                "actual": r.actual,
                "latency_ms": r.latency_ms,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            }
            for r in records
        ]

    async def flush(self) -> int:
        """Flush the current buffer to the database. Returns count flushed."""
        async with self._lock:
            return await self._do_flush()

    async def _do_flush(self) -> int:
        """Internal flush (caller must hold lock)."""
        if not self._buffer:
            return 0
        batch = self._buffer[:]
        self._buffer.clear()
        try:
            await self.db.insert_predictions(batch)
            logger.info("predictions_flushed", count=len(batch))
            return len(batch)
        except Exception:
            logger.exception("flush_failed", count=len(batch))
            self._buffer.extend(batch)
            return 0

    async def _flush_loop(self) -> None:
        """Periodic flush loop running in the background."""
        while self._running:
            await asyncio.sleep(self.flush_interval)
            await self.flush()

    @property
    def buffer_size(self) -> int:
        """Current number of buffered records."""
        return len(self._buffer)
