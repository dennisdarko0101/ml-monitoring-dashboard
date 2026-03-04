# Monitoring Guide

## Setting Up Model Monitoring

### 1. Log Predictions

```python
import httpx

async with httpx.AsyncClient() as client:
    await client.post("http://localhost:8000/api/v1/log/prediction", json={
        "model_name": "fraud-detector",
        "model_version": "2.1.0",
        "input_data": {"amount": 150.0, "merchant": "online"},
        "prediction": "legitimate",
        "latency_ms": 45.2,
    })
```

### 2. Log Ground Truth (When Available)

```python
await client.post("http://localhost:8000/api/v1/log/ground-truth", json={
    "prediction_id": "abc-123",
    "actual_value": "fraud",
})
```

### 3. Check Metrics

```python
resp = await client.get("http://localhost:8000/api/v1/metrics/fraud-detector?window=1h")
metrics = resp.json()
# Returns: latency_p50, p95, p99, throughput_rps, error_rate
```

### 4. Monitor Drift

```python
resp = await client.get("http://localhost:8000/api/v1/drift/fraud-detector")
drift = resp.json()
# Returns: overall_drift_score, is_drifted, drifted_features
```

## Understanding Alerts

### Severity Levels

| Level | Action | Channels |
|-------|--------|----------|
| INFO | Monitor | Log |
| WARNING | Investigate | Log + Webhook |
| CRITICAL | Immediate action | Log + Webhook + Email |

### Built-in Rules

- **high_latency**: Fires when P99 > 500ms. 5-minute cooldown.
- **error_spike**: Fires when error rate > 5%. 3-minute cooldown.
- **drift_detected**: Fires when drift score > 0.5. 10-minute cooldown.
- **model_degradation**: Fires when accuracy trend is negative. 10-minute cooldown.

### Custom Rules

```python
from src.alerting.alert_manager import AlertManager, AlertSeverity

manager = AlertManager()
manager.add_rule(
    name="slow_model",
    metric="latency_p95",
    condition="gt",
    threshold=200.0,
    severity=AlertSeverity.WARNING,
    cooldown=120.0,
)
```

## Drift Detection Interpretation

### Numeric Features
- **KS p-value < 0.05**: Distribution has significantly changed
- **PSI > 0.2**: Major population shift
- **PSI 0.1-0.2**: Moderate shift, worth investigating

### Categorical Features
- **Chi-squared p-value < 0.05**: Category frequencies changed significantly
- **JS divergence > 0.1**: Notable distributional difference

### Model Drift
- **Concept drift**: The relationship between inputs and outputs has changed
- **Prediction drift**: Model is producing different output distributions
- **Performance degradation**: Accuracy is trending downward over time

## Reports

```python
# Daily report
resp = await client.get("/api/v1/report/fraud-detector?period=daily")

# Weekly report
resp = await client.get("/api/v1/report/fraud-detector?period=weekly")
```

Reports include summary stats, active alerts, drift status, and auto-generated recommendations.
