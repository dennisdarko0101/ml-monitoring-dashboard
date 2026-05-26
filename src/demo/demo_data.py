"""Generator for clearly-labeled synthetic DEMO data.

This produces realistic-looking but entirely synthetic monitoring data for a
fictional ``demo-fraud-classifier`` model so the dashboard charts (including the
data-drift heatmap, which is computed with the real drift detector) render with
content for demonstration and screenshots.

It is deliberately NOT real production telemetry. The dashboard surfaces a
prominent banner saying so. Generation is seeded, so the dashboard is stable
across requests/restarts (useful for screenshots).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.detectors.data_drift import DataDriftDetector, DriftReport

DEMO_MODEL = "demo-fraud-classifier"
DEMO_BANNER = (
    "DEMO DATA — synthetic sample generated for demonstration and screenshots. "
    "These are not real production metrics."
)

_SEED = 42


@dataclass
class DemoData:
    """A bundle of synthetic series + a real drift report for the demo model."""

    model_name: str
    timestamps: list[str]
    latency_p50: list[float]
    latency_p95: list[float]
    latency_p99: list[float]
    throughput_rps: list[float]
    error_rate: list[float]
    pred_labels: list[str]
    pred_counts: list[int]
    drift: DriftReport


_CACHE: DemoData | None = None


def _timestamps(hours: int) -> list[str]:
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return [
        (now - timedelta(hours=hours - 1 - i)).strftime("%Y-%m-%d %H:%M")
        for i in range(hours)
    ]


def _drift_report(rng: np.random.Generator) -> DriftReport:
    """Compute a real drift report from synthetic reference vs current data.

    Several features are deliberately shifted so the heatmap shows a spread of
    drift scores rather than all-zeros.
    """
    n = 2000
    reference = pd.DataFrame(
        {
            "transaction_amount": rng.normal(52, 16, n),
            "account_age_days": rng.normal(420, 110, n),
            "items_per_order": rng.poisson(3.0, n).astype(float),
            "session_minutes": rng.gamma(2.0, 5.0, n),
            "device_type": rng.choice(
                ["mobile", "desktop", "tablet"], n, p=[0.55, 0.40, 0.05]
            ),
            "merchant_category": rng.choice(
                ["retail", "travel", "food", "digital"], n, p=[0.45, 0.20, 0.25, 0.10]
            ),
        }
    )
    current = pd.DataFrame(
        {
            "transaction_amount": rng.normal(64, 22, n),  # clear numeric drift
            "account_age_days": rng.normal(420, 110, n),  # stable (same as reference)
            "items_per_order": rng.poisson(3.0, n).astype(float),  # stable
            "session_minutes": rng.gamma(2.4, 6.2, n),  # moderate drift
            "device_type": rng.choice(
                ["mobile", "desktop", "tablet"], n, p=[0.72, 0.23, 0.05]
            ),  # categorical shift toward mobile
            "merchant_category": rng.choice(
                ["retail", "travel", "food", "digital"], n, p=[0.34, 0.18, 0.27, 0.21]
            ),  # categorical shift
        }
    )
    detector = DataDriftDetector()
    detector.set_reference(reference)
    return detector.detect(current)


def _build() -> DemoData:
    rng = np.random.default_rng(_SEED)
    ts = _timestamps(24)
    n = len(ts)

    # Latency percentiles (ms) with a congestion bump in the early evening.
    base = 42 + 6 * np.sin(np.linspace(0, np.pi, n))
    bump = np.zeros(n)
    bump[17:21] = [22, 58, 38, 16]
    p50 = base + rng.normal(0, 1.5, n)
    p95 = p50 * 2.1 + rng.normal(0, 4, n) + bump * 0.8
    p99 = p50 * 3.0 + rng.normal(0, 6, n) + bump

    # Throughput (req/s) following a daily usage curve.
    rps = 70 + 25 * np.sin(np.linspace(0, 2 * np.pi, n)) + rng.normal(0, 3, n)
    rps = np.clip(rps, 5, None)

    # Error rate with a spike above the 5% alert threshold during the bump.
    err = 0.008 + rng.normal(0, 0.0015, n)
    err = np.clip(err, 0.0, None)
    err[18] = 0.071
    err[19] = 0.053

    drift = _drift_report(rng)

    return DemoData(
        model_name=DEMO_MODEL,
        timestamps=ts,
        latency_p50=[round(float(x), 2) for x in p50],
        latency_p95=[round(float(x), 2) for x in p95],
        latency_p99=[round(float(x), 2) for x in p99],
        throughput_rps=[round(float(x), 2) for x in rps],
        error_rate=[round(float(x), 4) for x in err],
        pred_labels=["legitimate", "fraud"],
        pred_counts=[9180, 820],
        drift=drift,
    )


def get_demo_data() -> DemoData:
    """Return the cached demo data bundle, building it on first use."""
    global _CACHE
    if _CACHE is None:
        _CACHE = _build()
    return _CACHE
