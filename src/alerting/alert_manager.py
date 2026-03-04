"""Rule-based alert management with cooldowns and acknowledgment."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from src.storage.database import Database
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AlertRule:
    """A monitoring rule that triggers alerts when conditions are met."""

    name: str
    metric: str
    condition: str  # "gt", "lt", "gte", "lte", "eq"
    threshold: float
    severity: AlertSeverity = AlertSeverity.WARNING
    cooldown_seconds: float = 300.0
    message_template: str = ""
    _last_fired: float = 0.0


@dataclass
class Alert:
    """A triggered alert."""

    id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    acknowledged: bool = False
    metric_value: float = 0.0


CONDITION_OPS: dict[str, Callable[[float, float], bool]] = {
    "gt": lambda v, t: v > t,
    "lt": lambda v, t: v < t,
    "gte": lambda v, t: v >= t,
    "lte": lambda v, t: v <= t,
    "eq": lambda v, t: abs(v - t) < 1e-10,
}


class AlertManager:
    """Manages alert rules, evaluates conditions, tracks alerts."""

    def __init__(self, db: Database | None = None) -> None:
        self.db = db
        self._rules: dict[str, AlertRule] = {}
        self._active_alerts: list[Alert] = []
        self._setup_builtin_rules()

    def _setup_builtin_rules(self) -> None:
        """Register default monitoring rules."""
        self.add_rule(
            name="high_latency",
            metric="latency_p99",
            condition="gt",
            threshold=500.0,
            severity=AlertSeverity.WARNING,
            cooldown=300.0,
            message_template="P99 latency {value}ms exceeds threshold {threshold}ms",
        )
        self.add_rule(
            name="error_spike",
            metric="error_rate",
            condition="gt",
            threshold=0.05,
            severity=AlertSeverity.CRITICAL,
            cooldown=180.0,
            message_template="Error rate {value:.2%} exceeds threshold {threshold:.2%}",
        )
        self.add_rule(
            name="drift_detected",
            metric="drift_score",
            condition="gt",
            threshold=0.5,
            severity=AlertSeverity.WARNING,
            cooldown=600.0,
            message_template="Drift score {value:.4f} exceeds threshold {threshold:.4f}",
        )
        self.add_rule(
            name="model_degradation",
            metric="accuracy_trend",
            condition="lt",
            threshold=-0.01,
            severity=AlertSeverity.CRITICAL,
            cooldown=600.0,
            message_template="Model accuracy declining: trend {value:.4f}",
        )

    def add_rule(
        self,
        name: str,
        metric: str,
        condition: str,
        threshold: float,
        severity: AlertSeverity = AlertSeverity.WARNING,
        cooldown: float = 300.0,
        message_template: str = "",
    ) -> None:
        """Register or update an alerting rule."""
        if condition not in CONDITION_OPS:
            raise ValueError(f"Unknown condition '{condition}'. Use: {list(CONDITION_OPS)}")
        template = message_template or f"{metric} {condition} {threshold}: value={{value}}"
        self._rules[name] = AlertRule(
            name=name,
            metric=metric,
            condition=condition,
            threshold=threshold,
            severity=severity,
            cooldown_seconds=cooldown,
            message_template=template,
        )

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name."""
        return self._rules.pop(name, None) is not None

    def check_all(self, current_metrics: dict[str, float]) -> list[Alert]:
        """Evaluate all rules against current metrics. Returns newly fired alerts."""
        now = time.time()
        fired: list[Alert] = []

        for rule in self._rules.values():
            value = current_metrics.get(rule.metric)
            if value is None:
                continue

            check_fn = CONDITION_OPS[rule.condition]
            if not check_fn(value, rule.threshold):
                continue

            if now - rule._last_fired < rule.cooldown_seconds:
                continue

            alert = Alert(
                id=str(uuid.uuid4()),
                rule_name=rule.name,
                severity=rule.severity,
                message=rule.message_template.format(
                    value=value, threshold=rule.threshold
                ),
                timestamp=now,
                metric_value=value,
            )
            rule._last_fired = now
            self._active_alerts.append(alert)
            fired.append(alert)
            logger.warning(
                "alert_fired",
                rule=rule.name,
                severity=rule.severity.value,
                value=value,
            )

        return fired

    def acknowledge(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged."""
        for alert in self._active_alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def get_active_alerts(self) -> list[Alert]:
        """Return all unacknowledged alerts."""
        return [a for a in self._active_alerts if not a.acknowledged]

    def get_all_alerts(self) -> list[Alert]:
        """Return all alerts including acknowledged ones."""
        return list(self._active_alerts)

    def clear_acknowledged(self) -> int:
        """Remove acknowledged alerts from memory. Returns count removed."""
        before = len(self._active_alerts)
        self._active_alerts = [a for a in self._active_alerts if not a.acknowledged]
        return before - len(self._active_alerts)

    @property
    def rules(self) -> dict[str, AlertRule]:
        return dict(self._rules)
