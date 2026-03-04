# Architecture

## Overview

The ML Monitoring Dashboard follows a four-stage pipeline architecture:

```
Collect → Detect → Alert → Visualize
```

## Component Map

### Collection Layer (`src/collectors/`)
- **PredictionLogger** — Buffers prediction records, flushes in batches to SQLite
- **MetricsCollector** — Maintains sliding windows (1h/24h/7d) for latency, throughput, errors
- **SystemMonitor** — Captures CPU, memory, GPU, disk metrics

### Detection Layer (`src/detectors/`)
- **DataDriftDetector** — Statistical tests per feature (KS, PSI, Wasserstein, Chi-squared, JS divergence)
- **ModelDriftDetector** — Concept drift (Page-Hinkley, ADWIN), prediction distribution shift, accuracy trend analysis
- **AnomalyDetector** — Z-score, IQR, rolling deviation for operational anomalies

### Alerting Layer (`src/alerting/`)
- **AlertManager** — Rule engine with condition evaluation, cooldowns, acknowledgment
- **Notifier** — Multi-channel dispatch (structured log, webhook, email placeholder)

### Visualization Layer (`src/dashboard/`)
- **ChartGenerator** — Plotly JSON chart generation (7 chart types)
- **ReportGenerator** — Daily/weekly reports with auto-generated recommendations

### API Layer (`src/api/`)
- **FastAPI app** — REST endpoints, WebSocket live stream, Prometheus /metrics

### Storage Layer (`src/storage/`)
- **Database** — Async SQLAlchemy + aiosqlite with 5 tables (predictions, metrics, alerts, experiments, models)

## Data Flow

1. Client sends prediction to `POST /api/v1/log/prediction`
2. PredictionLogger buffers the record, MetricsCollector records latency
3. Periodic drift detection runs against accumulated predictions
4. AlertManager evaluates rules against current metrics
5. Notifier dispatches alerts to configured channels
6. Dashboard endpoints aggregate metrics, charts, and alerts
7. WebSocket streams live system metrics to connected clients

## Design Decisions

- **Buffered writes** — PredictionLogger batches inserts for high throughput (configurable batch size and flush interval)
- **Sliding windows** — In-memory deques for fast percentile computation without DB queries
- **SQLite default** — Zero-config development; swap to PostgreSQL via DB_URL for production
- **Prometheus integration** — Native histograms and counters for infrastructure monitoring tooling
- **Stateless detectors** — Drift detectors accept data directly; state management is external
