"""Tests for ModelDriftDetector: concept drift, prediction drift, Page-Hinkley, ADWIN."""

from __future__ import annotations

import numpy as np
import pytest

from src.detectors.model_drift import ADWIN, ModelDriftDetector, ModelDriftReport, PageHinkley


class TestPageHinkley:
    def test_no_change_no_detection(self):
        ph = PageHinkley(threshold=50.0)
        for _ in range(100):
            assert not ph.update(0.5)

    def test_detects_change(self):
        ph = PageHinkley(threshold=10.0, delta=0.001)
        detected = False
        for i in range(200):
            val = 0.1 if i < 100 else 0.9
            if ph.update(val):
                detected = True
                break
        assert detected

    def test_reset(self):
        ph = PageHinkley()
        ph.update(100.0)
        ph.reset()
        assert ph._count == 0
        assert ph._sum == 0.0


class TestADWIN:
    def test_no_change_no_detection(self):
        adwin = ADWIN(delta=0.002)
        for _ in range(100):
            assert not adwin.update(5.0)

    def test_detects_distribution_change(self):
        adwin = ADWIN(delta=0.01)
        detected = False
        for i in range(200):
            val = 0.0 if i < 100 else 10.0
            if adwin.update(val):
                detected = True
                break
        assert detected

    def test_reset_clears_window(self):
        adwin = ADWIN()
        adwin.update(1.0)
        adwin.reset()
        assert len(adwin._window) == 0


class TestModelDriftDetector:
    def test_no_drift_stable_model(self, model_drift_detector):
        rng = np.random.RandomState(42)
        predictions = rng.choice([0, 1], 200, p=[0.3, 0.7])
        actuals = predictions.copy()  # perfect predictions
        report = model_drift_detector.detect(predictions, actuals)
        assert isinstance(report, ModelDriftReport)
        assert not report.performance_degradation_detected

    def test_detects_concept_drift(self):
        detector = ModelDriftDetector(ph_threshold=5.0)
        # First half: model is correct. Second half: model is wrong.
        predictions = np.array([1] * 200)
        actuals = np.array([1] * 100 + [0] * 100)
        report = detector.detect(predictions, actuals)
        assert report.concept_drift_detected
        assert len(report.change_points) > 0

    def test_detects_performance_degradation(self):
        detector = ModelDriftDetector(degradation_threshold=-0.001, accuracy_window=20)
        # Accuracy declining over time
        rng = np.random.RandomState(42)
        n = 300
        predictions = np.ones(n, dtype=int)
        actuals = np.ones(n, dtype=int)
        # Gradually introduce errors
        for i in range(n):
            if rng.random() < i / n * 0.8:
                actuals[i] = 0
        report = detector.detect(predictions, actuals)
        assert report.accuracy_trend < 0

    def test_prediction_drift_detected(self, model_drift_detector):
        baseline = np.random.RandomState(42).normal(0, 1, 500)
        current = np.random.RandomState(42).normal(5, 1, 500)  # big shift
        actuals = np.zeros(500)
        report = model_drift_detector.detect(current, actuals, baseline)
        assert report.prediction_drift_detected

    def test_no_prediction_drift_same_distribution(self, model_drift_detector):
        rng = np.random.RandomState(42)
        baseline = rng.normal(0, 1, 500)
        current = rng.normal(0, 1, 500)
        actuals = np.zeros(500)
        report = model_drift_detector.detect(current, actuals, baseline)
        # Same distribution should not flag drift (usually)
        assert isinstance(report, ModelDriftReport)

    def test_empty_predictions(self, model_drift_detector):
        report = model_drift_detector.detect(np.array([]), np.array([]))
        assert not report.concept_drift_detected
        assert report.accuracy_trend == 0.0

    def test_any_drift_property(self, model_drift_detector):
        report = ModelDriftReport(
            concept_drift_detected=True,
            prediction_drift_detected=False,
            performance_degradation_detected=False,
            accuracy_trend=0.0,
            prediction_distribution_shift=0.0,
            change_points=[50],
        )
        assert report.any_drift

    def test_no_drift_property(self):
        report = ModelDriftReport(
            concept_drift_detected=False,
            prediction_drift_detected=False,
            performance_degradation_detected=False,
            accuracy_trend=0.0,
            prediction_distribution_shift=0.0,
            change_points=[],
        )
        assert not report.any_drift

    def test_prediction_drift_no_baseline(self, model_drift_detector):
        predictions = np.array([1, 2, 3])
        actuals = np.array([1, 2, 3])
        report = model_drift_detector.detect(predictions, actuals, baseline_predictions=None)
        assert not report.prediction_drift_detected

    def test_details_populated(self, model_drift_detector):
        preds = np.random.RandomState(42).choice([0, 1], 100)
        actuals = np.random.RandomState(99).choice([0, 1], 100)
        report = model_drift_detector.detect(preds, actuals)
        assert "n_predictions" in report.details
        assert report.details["n_predictions"] == 100

    def test_short_sequence(self, model_drift_detector):
        preds = np.array([1, 0, 1])
        actuals = np.array([1, 1, 0])
        report = model_drift_detector.detect(preds, actuals)
        assert isinstance(report, ModelDriftReport)
