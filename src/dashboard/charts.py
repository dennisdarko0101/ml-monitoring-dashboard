"""Plotly chart generation for the ML monitoring dashboard."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ChartGenerator:
    """Generates Plotly charts as JSON for frontend rendering."""

    @staticmethod
    def latency_timeseries(
        timestamps: list[str],
        p50: list[float],
        p95: list[float],
        p99: list[float],
    ) -> dict[str, Any]:
        """Generate a latency percentiles time series chart."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timestamps, y=p50, mode="lines", name="P50"))
        fig.add_trace(go.Scatter(x=timestamps, y=p95, mode="lines", name="P95"))
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=p99, mode="lines", name="P99",
                line=dict(color="red", dash="dash"),
            )
        )
        fig.update_layout(
            title="Prediction Latency Over Time",
            xaxis_title="Time",
            yaxis_title="Latency (ms)",
            template="plotly_white",
        )
        return fig.to_plotly_json()

    @staticmethod
    def throughput_chart(
        timestamps: list[str], rps: list[float]
    ) -> dict[str, Any]:
        """Generate a throughput (requests/sec) chart."""
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=rps, mode="lines+markers",
                name="Throughput", fill="tozeroy",
                line=dict(color="green"),
            )
        )
        fig.update_layout(
            title="Request Throughput",
            xaxis_title="Time",
            yaxis_title="Requests/sec",
            template="plotly_white",
        )
        return fig.to_plotly_json()

    @staticmethod
    def error_rate_chart(
        timestamps: list[str], error_rates: list[float], threshold: float = 0.05
    ) -> dict[str, Any]:
        """Generate error rate chart with threshold line."""
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=error_rates, mode="lines",
                name="Error Rate", line=dict(color="red"),
            )
        )
        fig.add_hline(
            y=threshold, line_dash="dash", line_color="orange",
            annotation_text=f"Threshold ({threshold:.0%})",
        )
        fig.update_layout(
            title="Error Rate",
            xaxis_title="Time",
            yaxis_title="Error Rate",
            template="plotly_white",
        )
        return fig.to_plotly_json()

    @staticmethod
    def drift_heatmap(
        features: list[str], drift_scores: list[float]
    ) -> dict[str, Any]:
        """Generate a heatmap of per-feature drift scores."""
        scores_2d = np.array(drift_scores).reshape(1, -1)
        fig = go.Figure(
            data=go.Heatmap(
                z=scores_2d.tolist(),
                x=features,
                y=["Drift Score"],
                colorscale="RdYlGn_r",
                zmin=0, zmax=1,
            )
        )
        fig.update_layout(
            title="Feature Drift Heatmap",
            template="plotly_white",
            height=250,
        )
        return fig.to_plotly_json()

    @staticmethod
    def prediction_distribution(
        labels: list[str], counts: list[int]
    ) -> dict[str, Any]:
        """Generate prediction label distribution chart."""
        fig = go.Figure(
            data=[go.Bar(x=labels, y=counts, marker_color="steelblue")]
        )
        fig.update_layout(
            title="Prediction Distribution",
            xaxis_title="Predicted Label",
            yaxis_title="Count",
            template="plotly_white",
        )
        return fig.to_plotly_json()

    @staticmethod
    def model_comparison_chart(
        model_names: list[str],
        metrics: dict[str, list[float]],
    ) -> dict[str, Any]:
        """Generate a grouped bar chart comparing models across metrics."""
        fig = go.Figure()
        for metric_name, values in metrics.items():
            fig.add_trace(go.Bar(name=metric_name, x=model_names, y=values))
        fig.update_layout(
            title="Model Comparison",
            barmode="group",
            template="plotly_white",
        )
        return fig.to_plotly_json()

    @staticmethod
    def confusion_matrix_chart(
        labels: list[str], matrix: list[list[int]]
    ) -> dict[str, Any]:
        """Generate a confusion matrix heatmap."""
        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=labels,
                y=labels,
                colorscale="Blues",
                text=matrix,
                texttemplate="%{text}",
            )
        )
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            template="plotly_white",
        )
        return fig.to_plotly_json()
