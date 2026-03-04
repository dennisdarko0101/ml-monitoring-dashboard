"""Tests for AnomalyDetector: latency, errors, volume anomalies."""

from __future__ import annotations

import numpy as np
import pytest

from src.detectors.anomaly_detector import Anomaly, AnomalyDetector, Severity


class TestLatencyAnomalies:
    def test_detects_outliers(self, anomaly_detector, normal_latencies):
        anomalies = anomaly_detector.detect_latency_anomalies(normal_latencies)
        assert len(anomalies) > 0
        values = [a.value for a in anomalies]
        assert 500.0 in values or 600.0 in values

    def test_no_anomaly_in_uniform_data(self, anomaly_detector):
        data = [50.0] * 100
        anomalies = anomaly_detector.detect_latency_anomalies(data)
        assert len(anomalies) == 0

    def test_too_few_data_points(self, anomaly_detector):
        anomalies = anomaly_detector.detect_latency_anomalies([10.0, 20.0])
        assert len(anomalies) == 0

    def test_anomaly_has_expected_fields(self, anomaly_detector, normal_latencies):
        anomalies = anomaly_detector.detect_latency_anomalies(normal_latencies)
        if anomalies:
            a = anomalies[0]
            assert a.metric == "latency"
            assert isinstance(a.expected_range, tuple)
            assert len(a.expected_range) == 2
            assert isinstance(a.severity, Severity)

    def test_with_timestamps(self, anomaly_detector, normal_latencies):
        ts = list(range(len(normal_latencies)))
        anomalies = anomaly_detector.detect_latency_anomalies(normal_latencies, timestamps=ts)
        assert all(isinstance(a.timestamp, float) for a in anomalies)


class TestErrorSpikes:
    def test_detects_error_spike(self, anomaly_detector, normal_error_counts):
        anomalies = anomaly_detector.detect_error_spikes(normal_error_counts)
        assert len(anomalies) > 0

    def test_no_spike_in_stable_errors(self, anomaly_detector):
        data = [2.0] * 100
        anomalies = anomaly_detector.detect_error_spikes(data)
        assert len(anomalies) == 0

    def test_spike_severity(self, anomaly_detector, normal_error_counts):
        anomalies = anomaly_detector.detect_error_spikes(normal_error_counts)
        severities = {a.severity for a in anomalies}
        assert len(severities) > 0

    def test_too_few_data_points(self, anomaly_detector):
        anomalies = anomaly_detector.detect_error_spikes([1.0])
        assert len(anomalies) == 0


class TestVolumeAnomalies:
    def test_detects_volume_drop(self, anomaly_detector):
        rng = np.random.RandomState(42)
        data = rng.normal(100, 5, 200).tolist()
        data[150] = 10.0  # sudden drop
        anomalies = anomaly_detector.detect_volume_anomalies(data)
        assert len(anomalies) > 0

    def test_detects_volume_surge(self, anomaly_detector):
        rng = np.random.RandomState(42)
        data = rng.normal(100, 5, 200).tolist()
        data[150] = 500.0  # sudden surge
        anomalies = anomaly_detector.detect_volume_anomalies(data)
        assert len(anomalies) > 0

    def test_stable_volume_no_anomalies(self, anomaly_detector):
        data = [100.0] * 100
        anomalies = anomaly_detector.detect_volume_anomalies(data)
        assert len(anomalies) == 0


class TestAnomalyDetectorConfig:
    def test_custom_z_threshold(self):
        # Very high threshold: nothing should be anomalous
        detector = AnomalyDetector(z_score_threshold=100.0)
        rng = np.random.RandomState(42)
        data = rng.normal(50, 10, 200).tolist()
        data[100] = 200.0
        anomalies = detector.detect_latency_anomalies(data)
        # With z=100, only extreme values flagged
        z_score_anomalies = [a for a in anomalies if a.method == "z_score"]
        assert len(z_score_anomalies) == 0

    def test_custom_iqr_multiplier(self):
        detector = AnomalyDetector(iqr_multiplier=0.5)  # very sensitive
        rng = np.random.RandomState(42)
        data = rng.normal(50, 10, 200).tolist()
        anomalies = detector.detect_latency_anomalies(data)
        # Should detect more anomalies with tight IQR
        assert len(anomalies) > 0

    def test_deduplication(self, anomaly_detector):
        # Same timestamp should be deduplicated
        rng = np.random.RandomState(42)
        data = rng.normal(50, 10, 200).tolist()
        data[100] = 500.0
        anomalies = anomaly_detector.detect_latency_anomalies(data)
        timestamps = [a.timestamp for a in anomalies]
        assert len(timestamps) == len(set(timestamps))
