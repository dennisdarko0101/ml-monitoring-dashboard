# Deployment Guide

## Local Development

```bash
pip install -e ".[dev]"
make run
```

## Docker Compose (Recommended)

```bash
make docker-up
```

Services:
- **API**: http://localhost:8000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## Production Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_MONITOR_DB_URL` | `sqlite+aiosqlite:///./ml_monitor.db` | Database URL |
| `ML_MONITOR_ALERT_WEBHOOK` | `` | Webhook URL for alerts |
| `ML_MONITOR_CHECK_INTERVAL_SECONDS` | `60` | Monitoring check interval |
| `ML_MONITOR_DRIFT_THRESHOLD` | `0.05` | KS test p-value threshold |
| `ML_MONITOR_LATENCY_P99_THRESHOLD` | `500.0` | P99 latency alert threshold (ms) |
| `ML_MONITOR_ERROR_RATE_THRESHOLD` | `0.05` | Error rate alert threshold |
| `ML_MONITOR_LOG_LEVEL` | `INFO` | Logging level |

### PostgreSQL (Production)

```bash
export ML_MONITOR_DB_URL="postgresql+asyncpg://user:pass@host:5432/mlmonitor"
pip install asyncpg
```

### Scaling

- Run multiple API workers: `uvicorn src.api.main:app --workers 4`
- Use Redis for shared metrics state across workers
- Deploy behind nginx/traefik for load balancing
- Use managed PostgreSQL for persistence

## Grafana Setup

1. Add Prometheus data source: `http://prometheus:9090`
2. Import dashboard from `docker/grafana-dashboard.json` (or create custom)
3. Key panels: prediction latency histogram, error rate gauge, active models count
