"""Tests for AlertManager and Notifier: rules, cooldown, acknowledge, notify."""

from __future__ import annotations

import time

import pytest

from src.alerting.alert_manager import Alert, AlertManager, AlertRule, AlertSeverity
from src.alerting.notifier import Notifier


class TestAlertManager:
    def test_builtin_rules_exist(self, alert_manager):
        rules = alert_manager.rules
        assert "high_latency" in rules
        assert "error_spike" in rules
        assert "drift_detected" in rules
        assert "model_degradation" in rules

    def test_add_custom_rule(self, alert_manager):
        alert_manager.add_rule(
            name="custom_rule",
            metric="custom_metric",
            condition="gt",
            threshold=10.0,
        )
        assert "custom_rule" in alert_manager.rules

    def test_remove_rule(self, alert_manager):
        assert alert_manager.remove_rule("high_latency")
        assert "high_latency" not in alert_manager.rules

    def test_remove_nonexistent_rule(self, alert_manager):
        assert not alert_manager.remove_rule("nonexistent")

    def test_invalid_condition_raises(self, alert_manager):
        with pytest.raises(ValueError, match="Unknown condition"):
            alert_manager.add_rule("bad", "m", "invalid_op", 1.0)

    def test_check_all_fires_alert(self, alert_manager):
        metrics = {"latency_p99": 600.0}  # exceeds threshold
        alerts = alert_manager.check_all(metrics)
        assert len(alerts) >= 1
        assert any(a.rule_name == "high_latency" for a in alerts)

    def test_check_all_no_fire_below_threshold(self, alert_manager):
        metrics = {"latency_p99": 100.0}  # below threshold
        alerts = alert_manager.check_all(metrics)
        high_lat = [a for a in alerts if a.rule_name == "high_latency"]
        assert len(high_lat) == 0

    def test_cooldown_prevents_repeated_fire(self, alert_manager):
        metrics = {"latency_p99": 600.0}
        alert_manager.check_all(metrics)
        # Immediate re-check should be suppressed by cooldown
        alerts2 = alert_manager.check_all(metrics)
        high_lat = [a for a in alerts2 if a.rule_name == "high_latency"]
        assert len(high_lat) == 0

    def test_acknowledge_alert(self, alert_manager):
        metrics = {"latency_p99": 600.0}
        alerts = alert_manager.check_all(metrics)
        alert_id = alerts[0].id
        assert alert_manager.acknowledge(alert_id)
        active = alert_manager.get_active_alerts()
        assert all(a.id != alert_id for a in active)

    def test_acknowledge_nonexistent(self, alert_manager):
        assert not alert_manager.acknowledge("fake-id")

    def test_get_active_alerts(self, alert_manager):
        metrics = {"latency_p99": 600.0, "error_rate": 0.1}
        alert_manager.check_all(metrics)
        active = alert_manager.get_active_alerts()
        assert len(active) > 0
        assert all(not a.acknowledged for a in active)

    def test_get_all_alerts(self, alert_manager):
        metrics = {"latency_p99": 600.0}
        alerts = alert_manager.check_all(metrics)
        alert_manager.acknowledge(alerts[0].id)
        all_alerts = alert_manager.get_all_alerts()
        assert len(all_alerts) >= 1

    def test_clear_acknowledged(self, alert_manager):
        metrics = {"latency_p99": 600.0}
        alerts = alert_manager.check_all(metrics)
        alert_manager.acknowledge(alerts[0].id)
        removed = alert_manager.clear_acknowledged()
        assert removed >= 1

    def test_multiple_rules_fire(self, alert_manager):
        metrics = {"latency_p99": 600.0, "error_rate": 0.1}
        alerts = alert_manager.check_all(metrics)
        rule_names = {a.rule_name for a in alerts}
        assert "high_latency" in rule_names
        assert "error_spike" in rule_names

    def test_alert_severity(self, alert_manager):
        metrics = {"error_rate": 0.1}
        alerts = alert_manager.check_all(metrics)
        error_alerts = [a for a in alerts if a.rule_name == "error_spike"]
        assert error_alerts[0].severity == AlertSeverity.CRITICAL


class TestNotifier:
    def test_notify_log(self):
        notifier = Notifier()
        alert = Alert(
            id="test-1",
            rule_name="test_rule",
            severity=AlertSeverity.INFO,
            message="Test message",
            timestamp=time.time(),
        )
        assert notifier.notify_log(alert)

    async def test_notify_webhook_no_url(self):
        notifier = Notifier()
        alert = Alert(
            id="test-1",
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            message="Test message",
            timestamp=time.time(),
        )
        result = await notifier.notify_webhook(alert, "")
        assert not result

    async def test_notify_email_no_recipients(self):
        notifier = Notifier()
        alert = Alert(
            id="test-1",
            rule_name="test_rule",
            severity=AlertSeverity.CRITICAL,
            message="Test message",
            timestamp=time.time(),
        )
        result = await notifier.notify_email(alert, [])
        assert not result

    async def test_notify_email_with_recipients(self):
        notifier = Notifier()
        alert = Alert(
            id="test-1",
            rule_name="test_rule",
            severity=AlertSeverity.CRITICAL,
            message="Test message",
            timestamp=time.time(),
        )
        result = await notifier.notify_email(alert, ["user@example.com"])
        assert result

    def test_configure_channels(self):
        notifier = Notifier()
        notifier.configure_channels(AlertSeverity.INFO, ["log", "webhook"])
        assert notifier._channels[AlertSeverity.INFO] == ["log", "webhook"]

    def test_configure_invalid_channel(self):
        notifier = Notifier()
        with pytest.raises(ValueError, match="Unknown channel"):
            notifier.configure_channels(AlertSeverity.INFO, ["sms"])

    async def test_notify_dispatches(self):
        notifier = Notifier()
        alert = Alert(
            id="test-1",
            rule_name="test_rule",
            severity=AlertSeverity.INFO,
            message="Test message",
            timestamp=time.time(),
        )
        results = await notifier.notify(alert)
        assert results["log"] is True
