# ML Monitoring Dashboard

A production ML observability platform that monitors model performance, detects drift, tracks experiments, and provides real-time dashboards. Built with FastAPI, SQLAlchemy, and Plotly.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Collect    │────▶│    Detect    │────▶│    Alert    │────▶│  Visualize   │
│              │     │              │     │              │     │              │
│ Predictions  │     │ Data Drift   │     │ Rules Engine │     │ Plotly Charts│
│ Latency      │     │ Model Drift  │     │ Cooldowns    │     │ Reports      │
│ Errors       │     │ Anomalies    │     │ Webhooks     │     │ WebSocket    │
│ System       │     │              │     │ Email        │     │ Prometheus   │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
```

## Monitoring Capabilities

- **Prediction Logging** — High-throughput buffered writes with async batch inserts
- **Metrics Collection** — Latency percentiles (P50/P95/P99), throughput, error rates with sliding windows (1h/24h/7d)
- **Data Drift Detection** — KS test, PSI, Wasserstein distance (numeric), Chi-squared, Jensen-Shannon divergence (categorical)
- **Model Drift Detection** — Concept drift (Page-Hinkley, ADWIN), prediction distribution shift, performance degradation trends
- **Anomaly Detection** — Z-score, IQR, rolling average deviation for latency/errors/volume
- **Alerting** — Rule-based with configurable thresholds, cooldowns, severity levels, multi-channel notifications
- **Dashboards** — Real-time Plotly charts, daily/weekly reports (HTML/JSON/Markdown), WebSocket live streaming
- **System Monitoring** — CPU, memory, GPU, disk usage tracking

## Drift Detection Methodology

| Method | Feature Type | Detects |
|--------|-------------|---------|
| Kolmogorov-Smirnov | Numeric | Distribution shape changes |
| Population Stability Index | Numeric | Distribution shift magnitude |
| Wasserstein Distance | Numeric | Earth mover's distance between distributions |
| Chi-Squared Test | Categorical | Category frequency changes |
| Jensen-Shannon Divergence | Categorical | Symmetric distribution divergence |
| Page-Hinkley | Time Series | Change points in accuracy stream |
| ADWIN | Time Series | Adaptive window change detection |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run
make run
# → API at http://localhost:8000
# → Docs at http://localhost:8000/docs

# Docker
make docker-up
# → API at :8000, Prometheus at :9090, Grafana at :3000
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/log/prediction` | Log a model prediction |
| POST | `/api/v1/log/ground-truth` | Log delayed ground truth |
| GET | `/api/v1/metrics/{model}` | Get metrics summary |
| GET | `/api/v1/drift/{model}` | Get drift report |
| GET | `/api/v1/alerts` | List active alerts |
| POST | `/api/v1/alerts/{id}/acknowledge` | Acknowledge alert |
| GET | `/api/v1/dashboard/{model}` | Full dashboard data |
| GET | `/api/v1/report/{model}` | Generate report |
| GET | `/api/v1/models` | List monitored models |
| GET | `/api/v1/system` | System health |
| WS | `/ws/live` | Live metrics stream |
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |

## Alert Configuration

Built-in rules:
- `high_latency` — P99 > 500ms (warning)
- `error_spike` — Error rate > 5% (critical)
- `drift_detected` — Drift score > 0.5 (warning)
- `model_degradation` — Accuracy trend declining (critical)

Custom rules via API or environment:
```python
alert_manager.add_rule(
    name="custom_rule",
    metric="latency_p99",
    condition="gt",
    threshold=200.0,
    severity=AlertSeverity.WARNING,
    cooldown=60.0,
)
```

## Dashboard Screenshots

The dashboard provides:
- **Latency Time Series** — P50/P95/P99 percentile trends over time
- **Throughput Chart** — Requests per second with fill area
- **Error Rate** — Error percentage with threshold line overlay
- **Drift Heatmap** — Per-feature drift scores as a color-coded heatmap
- **Prediction Distribution** — Bar chart of prediction label frequencies
- **Model Comparison** — Grouped bar chart comparing models across metrics
- **Confusion Matrix** — Heatmap with annotated cell counts

## Tech Stack

- **API**: FastAPI + Uvicorn
- **Database**: SQLAlchemy + aiosqlite (SQLite default, PostgreSQL-ready)
- **Visualization**: Plotly
- **Monitoring**: Prometheus + Grafana
- **ML/Stats**: NumPy, SciPy, scikit-learn, Pandas
- **Testing**: pytest + pytest-asyncio (140+ tests)

## Development

```bash
make dev          # Install with dev deps
make test         # Run all tests
make coverage     # Run tests with coverage report
make lint         # Run linter
make typecheck    # Run type checker
```

## License

MIT
