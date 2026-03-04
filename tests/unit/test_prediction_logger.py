"""Tests for PredictionLogger: buffering, flushing, ground truth."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from src.collectors.prediction_logger import PredictionLogger
from src.storage.database import Database


@pytest.fixture
def logger(db, prediction_logger):
    return prediction_logger


class TestPredictionLogger:
    async def test_log_adds_to_buffer(self, logger):
        await logger.log("model_a", "1.0", {"x": 1}, 0.95, 25.0)
        assert logger.buffer_size == 1

    async def test_log_multiple(self, logger):
        for i in range(5):
            await logger.log("model_a", "1.0", {"x": i}, float(i), 10.0 + i)
        assert logger.buffer_size == 5

    async def test_flush_writes_to_db(self, logger):
        for i in range(3):
            await logger.log("model_a", "1.0", {"x": i}, float(i), 10.0)
        count = await logger.flush()
        assert count == 3
        assert logger.buffer_size == 0

    async def test_flush_empty_buffer_returns_zero(self, logger):
        count = await logger.flush()
        assert count == 0

    async def test_auto_flush_on_batch_size(self, db):
        logger = PredictionLogger(db)
        logger.batch_size = 3
        for i in range(3):
            await logger.log("model_a", "1.0", {"x": i}, float(i), 10.0)
        # Buffer should have been auto-flushed
        assert logger.buffer_size == 0

    async def test_get_predictions_after_flush(self, logger):
        await logger.log("model_a", "1.0", {"x": 1}, 0.9, 20.0)
        await logger.flush()
        preds = await logger.get_predictions("model_a")
        assert len(preds) == 1
        assert preds[0]["model_name"] == "model_a"

    async def test_get_predictions_filters_by_model(self, logger):
        await logger.log("model_a", "1.0", {"x": 1}, 0.9, 20.0)
        await logger.log("model_b", "1.0", {"x": 2}, 0.8, 30.0)
        await logger.flush()
        preds = await logger.get_predictions("model_a")
        assert len(preds) == 1

    async def test_ground_truth_update(self, db, logger):
        await logger.log("model_a", "1.0", {"x": 1}, 0.9, 20.0)
        await logger.flush()
        preds = await logger.get_predictions("model_a")
        pred_id = preds[0]["id"]
        await logger.log_ground_truth(pred_id, 1.0)
        updated = await db.get_predictions("model_a")
        assert updated[0].actual is not None

    async def test_ground_truth_nonexistent_id(self, logger):
        # Should not raise, just log warning
        await logger.log_ground_truth("nonexistent-id", 1.0)

    async def test_input_hash_deterministic(self, logger):
        await logger.log("m", "1.0", {"a": 1, "b": 2}, 0.5, 10.0)
        await logger.log("m", "1.0", {"b": 2, "a": 1}, 0.5, 10.0)
        await logger.flush()
        preds = await logger.get_predictions("m")
        assert preds[0]["input_hash"] == preds[1]["input_hash"]

    async def test_start_stop_lifecycle(self, db):
        logger = PredictionLogger(db)
        await logger.start()
        await logger.log("m", "1.0", {"x": 1}, 0.5, 10.0)
        await logger.stop()
        # After stop, buffer should be flushed
        assert logger.buffer_size == 0

    async def test_prediction_serializes_complex_types(self, logger):
        await logger.log("m", "1.0", {"nested": {"a": [1, 2]}}, {"class": "A", "prob": 0.9}, 15.0)
        await logger.flush()
        preds = await logger.get_predictions("m")
        assert len(preds) == 1
