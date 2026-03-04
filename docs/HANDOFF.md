# Handoff Document

## Project Summary

ML Monitoring Dashboard is a production-ready ML observability platform. It provides end-to-end model monitoring: prediction logging, metrics aggregation, statistical drift detection, rule-based alerting, and real-time visualization.

## Repository Structure

```
ml-monitoring-dashboard/
├── src/
│   ├── api/            # FastAPI endpoints + Pydantic schemas
│   ├── collectors/     # Prediction logger, metrics collector, system monitor
│   ├── detectors/      # Data drift, model drift, anomaly detection
│   ├── alerting/       # Alert manager + notifier
│   ├── dashboard/      # Chart generation + report generation
│   ├── storage/        # SQLAlchemy database layer
│   ├── config/         # Settings (pydantic-settings)
│   └── utils/          # Structured logging
├── tests/
│   ├── unit/           # 100+ unit tests across all modules
│   └── integration/    # API and pipeline integration tests
├── docker/             # Dockerfile, docker-compose, Prometheus config
├── docs/               # Architecture, deployment, monitoring guides
└── .github/workflows/  # CI/CD pipelines
```

## Key Files

- `src/api/main.py` — FastAPI app with all endpoints
- `src/storage/database.py` — Database models and async CRUD
- `src/detectors/data_drift.py` — Statistical drift detection (KS, PSI, chi-squared)
- `src/alerting/alert_manager.py` — Rule engine with 4 built-in rules
- `src/collectors/prediction_logger.py` — Buffered async prediction logging

## Running

```bash
make dev     # Install
make run     # Start server
make test    # Run tests
```

## Extending

- **New drift method**: Add to `DataDriftDetector._detect_numeric()` or `_detect_categorical()`
- **New alert rule**: Call `alert_manager.add_rule()` or add to `_setup_builtin_rules()`
- **New chart type**: Add a static method to `ChartGenerator`
- **New notification channel**: Add method to `Notifier`, update `_channels` mapping
- **Production database**: Change `ML_MONITOR_DB_URL` to PostgreSQL + asyncpg

## Known Limitations

- Drift detection is stateless per request (no periodic background job yet)
- Email notifications are placeholder — integrate with SES/SendGrid for production
- WebSocket sends system metrics only — extend for custom metric streams
- Single-process sliding windows — use Redis for multi-worker deployments
