"""Alert notification channels: webhook, log, email (placeholder)."""

from __future__ import annotations

from typing import Any

import httpx

from src.alerting.alert_manager import Alert, AlertSeverity
from src.config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Notifier:
    """Dispatches alerts to configured notification channels."""

    def __init__(self) -> None:
        self._channels: dict[AlertSeverity, list[str]] = {
            AlertSeverity.INFO: ["log"],
            AlertSeverity.WARNING: ["log", "webhook"],
            AlertSeverity.CRITICAL: ["log", "webhook", "email"],
        }

    def configure_channels(
        self, severity: AlertSeverity, channels: list[str]
    ) -> None:
        """Override notification channels for a severity level."""
        valid = {"log", "webhook", "email"}
        for ch in channels:
            if ch not in valid:
                raise ValueError(f"Unknown channel '{ch}'. Valid: {valid}")
        self._channels[severity] = channels

    async def notify(self, alert: Alert) -> dict[str, bool]:
        """Send alert to all configured channels for its severity. Returns status per channel."""
        channels = self._channels.get(alert.severity, ["log"])
        results: dict[str, bool] = {}

        for channel in channels:
            if channel == "log":
                results["log"] = self.notify_log(alert)
            elif channel == "webhook":
                results["webhook"] = await self.notify_webhook(
                    alert, settings.ALERT_WEBHOOK
                )
            elif channel == "email":
                recipients = [
                    r.strip()
                    for r in settings.ALERT_EMAIL_RECIPIENTS.split(",")
                    if r.strip()
                ]
                results["email"] = await self.notify_email(alert, recipients)

        return results

    def notify_log(self, alert: Alert) -> bool:
        """Log the alert using structured logging."""
        logger.warning(
            "alert_notification",
            alert_id=alert.id,
            rule=alert.rule_name,
            severity=alert.severity.value,
            message=alert.message,
        )
        return True

    async def notify_webhook(self, alert: Alert, url: str) -> bool:
        """Send alert payload to a webhook URL."""
        if not url:
            logger.debug("webhook_skipped", reason="no_url_configured")
            return False

        payload = {
            "alert_id": alert.id,
            "rule": alert.rule_name,
            "severity": alert.severity.value,
            "message": alert.message,
            "timestamp": alert.timestamp,
            "metric_value": alert.metric_value,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
            logger.info("webhook_sent", alert_id=alert.id, status=resp.status_code)
            return True
        except httpx.HTTPError:
            logger.exception("webhook_failed", alert_id=alert.id, url=url)
            return False

    async def notify_email(self, alert: Alert, recipients: list[str]) -> bool:
        """Placeholder for email notifications.

        In production, integrate with SendGrid, SES, or SMTP.
        """
        if not recipients:
            return False

        logger.info(
            "email_notification_placeholder",
            alert_id=alert.id,
            recipients=recipients,
            subject=f"[{alert.severity.value.upper()}] {alert.rule_name}",
            body=alert.message,
        )
        # Placeholder: would send email here
        return True
