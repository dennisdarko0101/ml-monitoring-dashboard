"""Report generation: daily/weekly summaries with charts, alerts, and recommendations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from jinja2 import Template

from src.collectors.metrics_collector import MetricsCollector, MetricsSummary
from src.dashboard.charts import ChartGenerator
from src.detectors.data_drift import DriftReport
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Report:
    """A monitoring report."""

    model_name: str
    period: str
    generated_at: str
    summary_stats: dict[str, Any] = field(default_factory=dict)
    charts: dict[str, Any] = field(default_factory=dict)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    drift_status: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "model_name": self.model_name,
                "period": self.period,
                "generated_at": self.generated_at,
                "summary_stats": self.summary_stats,
                "alerts": self.alerts,
                "drift_status": self.drift_status,
                "recommendations": self.recommendations,
            },
            indent=2,
            default=str,
        )

    def to_markdown(self) -> str:
        lines = [
            f"# Monitoring Report: {self.model_name}",
            f"**Period:** {self.period}",
            f"**Generated:** {self.generated_at}",
            "",
            "## Summary",
        ]
        for k, v in self.summary_stats.items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

        if self.drift_status:
            lines.append("## Drift Status")
            lines.append(f"- **Overall Drifted:** {self.drift_status.get('is_drifted', False)}")
            for feat in self.drift_status.get("drifted_features", []):
                lines.append(f"  - {feat}")
            lines.append("")

        if self.alerts:
            lines.append("## Alerts")
            for a in self.alerts:
                lines.append(f"- [{a.get('severity', 'info').upper()}] {a.get('message', '')}")
            lines.append("")

        if self.recommendations:
            lines.append("## Recommendations")
            for r in self.recommendations:
                lines.append(f"- {r}")

        return "\n".join(lines)

    def to_html(self) -> str:
        template = Template(HTML_TEMPLATE)
        return template.render(report=self)


HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><title>Report: {{ report.model_name }}</title>
<style>body{font-family:sans-serif;margin:2rem;}
h1{color:#333;} .stat{margin:0.5rem 0;} .alert{padding:0.5rem;margin:0.3rem 0;border-left:3px solid #ccc;}
.critical{border-color:red;} .warning{border-color:orange;} .info{border-color:blue;}
</style></head><body>
<h1>Monitoring Report: {{ report.model_name }}</h1>
<p><strong>Period:</strong> {{ report.period }} | <strong>Generated:</strong> {{ report.generated_at }}</p>
<h2>Summary</h2>
{% for k, v in report.summary_stats.items() %}<div class="stat"><strong>{{ k }}:</strong> {{ v }}</div>{% endfor %}
<h2>Drift Status</h2>
<p>Drifted: {{ report.drift_status.get('is_drifted', False) }}</p>
{% for feat in report.drift_status.get('drifted_features', []) %}<div>- {{ feat }}</div>{% endfor %}
<h2>Alerts ({{ report.alerts|length }})</h2>
{% for a in report.alerts %}<div class="alert {{ a.get('severity', 'info') }}">{{ a.get('message', '') }}</div>{% endfor %}
<h2>Recommendations</h2>
<ul>{% for r in report.recommendations %}<li>{{ r }}</li>{% endfor %}</ul>
</body></html>"""


class ReportGenerator:
    """Generates daily and weekly monitoring reports."""

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        chart_generator: ChartGenerator | None = None,
    ) -> None:
        self.metrics = metrics_collector
        self.charts = chart_generator or ChartGenerator()

    def daily_report(
        self,
        model_name: str,
        date: datetime | None = None,
        drift_report: DriftReport | None = None,
        alerts: list[dict[str, Any]] | None = None,
    ) -> Report:
        """Generate a daily monitoring report."""
        summary = self.metrics.get_summary(model_name, "24h")
        return self._build_report(
            model_name=model_name,
            period="daily",
            summary=summary,
            drift_report=drift_report,
            alerts=alerts or [],
        )

    def weekly_report(
        self,
        model_name: str,
        drift_report: DriftReport | None = None,
        alerts: list[dict[str, Any]] | None = None,
    ) -> Report:
        """Generate a weekly monitoring report."""
        summary = self.metrics.get_summary(model_name, "7d")
        return self._build_report(
            model_name=model_name,
            period="weekly",
            summary=summary,
            drift_report=drift_report,
            alerts=alerts or [],
        )

    def _build_report(
        self,
        model_name: str,
        period: str,
        summary: MetricsSummary,
        drift_report: DriftReport | None,
        alerts: list[dict[str, Any]],
    ) -> Report:
        """Assemble a full report from components."""
        summary_stats = {
            "Latency P50": f"{summary.latency_p50}ms",
            "Latency P95": f"{summary.latency_p95}ms",
            "Latency P99": f"{summary.latency_p99}ms",
            "Throughput": f"{summary.throughput_rps} req/s",
            "Error Rate": f"{summary.error_rate:.2%}",
            "Total Predictions": summary.total_predictions,
        }

        drift_status: dict[str, Any] = {}
        if drift_report:
            drift_status = {
                "is_drifted": drift_report.is_drifted,
                "overall_score": drift_report.overall_drift_score,
                "drifted_features": drift_report.drifted_features,
            }

        recommendations = self._generate_recommendations(summary, drift_report)

        return Report(
            model_name=model_name,
            period=period,
            generated_at=datetime.now(timezone.utc).isoformat(),
            summary_stats=summary_stats,
            alerts=alerts,
            drift_status=drift_status,
            recommendations=recommendations,
        )

    @staticmethod
    def _generate_recommendations(
        summary: MetricsSummary, drift_report: DriftReport | None
    ) -> list[str]:
        """Auto-generate actionable recommendations based on metrics."""
        recs: list[str] = []

        if summary.latency_p99 > 500:
            recs.append(
                "P99 latency is high. Consider model optimization, caching, or scaling."
            )
        if summary.error_rate > 0.05:
            recs.append(
                "Error rate exceeds 5%. Investigate input validation and model robustness."
            )
        if summary.throughput_rps < 1.0 and summary.total_predictions > 0:
            recs.append(
                "Low throughput detected. Check for bottlenecks in the serving pipeline."
            )
        if drift_report and drift_report.is_drifted:
            recs.append(
                f"Data drift detected in {len(drift_report.drifted_features)} feature(s). "
                "Consider retraining on recent data."
            )
        if not recs:
            recs.append("All metrics within normal range. No action needed.")

        return recs
