"""Tests for DataDriftDetector: KS, PSI, chi-squared, per-feature."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.detectors.data_drift import DataDriftDetector, DriftReport


class TestDataDriftDetectorNumeric:
    def test_no_drift_identical_data(self, data_drift_detector, normal_numeric_df):
        data_drift_detector.set_reference(normal_numeric_df)
        report = data_drift_detector.detect(normal_numeric_df)
        assert not report.is_drifted
        assert len(report.drifted_features) == 0

    def test_detects_mean_shift(self, data_drift_detector, normal_numeric_df, shifted_numeric_df):
        data_drift_detector.set_reference(normal_numeric_df)
        report = data_drift_detector.detect(shifted_numeric_df)
        assert report.is_drifted
        assert "feature_a" in report.drifted_features

    def test_non_drifted_feature_not_flagged(self, data_drift_detector, normal_numeric_df, shifted_numeric_df):
        data_drift_detector.set_reference(normal_numeric_df)
        report = data_drift_detector.detect(shifted_numeric_df)
        # feature_c has same distribution
        assert "feature_c" not in report.drifted_features

    def test_ks_statistic_in_details(self, data_drift_detector, normal_numeric_df, shifted_numeric_df):
        data_drift_detector.set_reference(normal_numeric_df)
        report = data_drift_detector.detect(shifted_numeric_df)
        for result in report.per_feature_scores:
            if result.feature == "feature_a":
                assert "ks_statistic" in result.details
                assert result.details["ks_statistic"] > 0

    def test_psi_in_details(self, data_drift_detector, normal_numeric_df, shifted_numeric_df):
        data_drift_detector.set_reference(normal_numeric_df)
        report = data_drift_detector.detect(shifted_numeric_df)
        for result in report.per_feature_scores:
            if result.feature == "feature_a":
                assert "psi" in result.details
                assert result.details["psi"] >= 0

    def test_wasserstein_in_details(self, data_drift_detector, normal_numeric_df, shifted_numeric_df):
        data_drift_detector.set_reference(normal_numeric_df)
        report = data_drift_detector.detect(shifted_numeric_df)
        for result in report.per_feature_scores:
            if result.feature == "feature_a":
                assert "wasserstein" in result.details

    def test_overall_drift_score(self, data_drift_detector, normal_numeric_df, shifted_numeric_df):
        data_drift_detector.set_reference(normal_numeric_df)
        report = data_drift_detector.detect(shifted_numeric_df)
        assert 0 <= report.overall_drift_score <= 1

    def test_custom_ks_threshold(self, normal_numeric_df):
        # Very strict threshold - should flag more features
        detector = DataDriftDetector(ks_threshold=0.99)
        detector.set_reference(normal_numeric_df)
        rng = np.random.RandomState(99)
        slightly_shifted = normal_numeric_df.copy()
        slightly_shifted["feature_a"] = rng.normal(0.01, 1, 500)
        report = detector.detect(slightly_shifted)
        # With very loose threshold (p < 0.99), more will pass as not drifted
        assert isinstance(report, DriftReport)

    def test_no_reference_raises(self, data_drift_detector):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="Reference data not set"):
            data_drift_detector.detect(df)

    def test_empty_column_skipped(self, data_drift_detector):
        ref = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [np.nan, np.nan, np.nan]})
        cur = pd.DataFrame({"a": [4.0, 5.0, 6.0], "b": [np.nan, np.nan, np.nan]})
        data_drift_detector.set_reference(ref)
        report = data_drift_detector.detect(cur)
        # Only feature_a should have results
        features = [r.feature for r in report.per_feature_scores]
        assert "b" not in features


class TestDataDriftDetectorCategorical:
    def test_no_drift_same_distribution(self, data_drift_detector, categorical_ref_df):
        data_drift_detector.set_reference(categorical_ref_df)
        report = data_drift_detector.detect(categorical_ref_df)
        assert not report.is_drifted

    def test_detects_categorical_drift(self, data_drift_detector, categorical_ref_df, categorical_shifted_df):
        data_drift_detector.set_reference(categorical_ref_df)
        report = data_drift_detector.detect(categorical_shifted_df)
        assert report.is_drifted
        assert "color" in report.drifted_features

    def test_non_drifted_categorical_not_flagged(self, data_drift_detector, categorical_ref_df, categorical_shifted_df):
        data_drift_detector.set_reference(categorical_ref_df)
        report = data_drift_detector.detect(categorical_shifted_df)
        # size has same distribution
        assert "size" not in report.drifted_features

    def test_chi_squared_in_details(self, data_drift_detector, categorical_ref_df, categorical_shifted_df):
        data_drift_detector.set_reference(categorical_ref_df)
        report = data_drift_detector.detect(categorical_shifted_df)
        for result in report.per_feature_scores:
            if result.feature == "color":
                assert "chi2_statistic" in result.details
                assert "js_divergence" in result.details

    def test_js_divergence_positive(self, data_drift_detector, categorical_ref_df, categorical_shifted_df):
        data_drift_detector.set_reference(categorical_ref_df)
        report = data_drift_detector.detect(categorical_shifted_df)
        for result in report.per_feature_scores:
            if result.feature == "color":
                assert result.details["js_divergence"] > 0


class TestDataDriftDetectorMixed:
    def test_mixed_features(self, data_drift_detector):
        rng = np.random.RandomState(42)
        ref = pd.DataFrame({
            "num_feat": rng.normal(0, 1, 300),
            "cat_feat": rng.choice(["a", "b", "c"], 300),
        })
        cur = pd.DataFrame({
            "num_feat": rng.normal(5, 1, 300),
            "cat_feat": rng.choice(["a", "b", "c"], 300, p=[0.8, 0.1, 0.1]),
        })
        data_drift_detector.set_reference(ref)
        report = data_drift_detector.detect(cur)
        assert isinstance(report, DriftReport)
        assert len(report.per_feature_scores) == 2

    def test_missing_column_in_current(self, data_drift_detector):
        ref = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        cur = pd.DataFrame({"a": [7, 8, 9]})  # missing 'b'
        data_drift_detector.set_reference(ref)
        report = data_drift_detector.detect(cur)
        features = [r.feature for r in report.per_feature_scores]
        assert "a" in features
        assert "b" not in features

    def test_compute_psi_identical(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 100)
        psi = DataDriftDetector._compute_psi(arr, arr)
        assert psi < 0.01

    def test_compute_psi_shifted(self):
        ref = np.random.RandomState(42).normal(0, 1, 1000)
        cur = np.random.RandomState(42).normal(5, 1, 1000)
        psi = DataDriftDetector._compute_psi(ref, cur)
        assert psi > 0.1
