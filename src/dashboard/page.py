"""Server-rendered HTML dashboard page that renders the Plotly charts.

The API otherwise exposes chart data as JSON; this module assembles those
figures into a single self-contained HTML page (Plotly.js from CDN) so the
dashboard can be opened in a browser and screenshotted. It is fed by the
synthetic demo data bundle, with a prominent banner marking it as such.
"""

from __future__ import annotations

import json

import plotly.utils

from src.dashboard.charts import ChartGenerator
from src.demo.demo_data import DEMO_BANNER, DemoData, get_demo_data


def _fig_json(figure: dict) -> str:
    """Serialize a Plotly figure dict (numpy-safe) for embedding in the page."""
    return json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)


def _prep(figure: dict) -> dict:
    """Presentation-only layout normalization so figures fill their containers.

    Lets each figure size to its 320px container (no fixed height/width) and
    reserves margin above the plot area so the title never overlaps the traces.
    """
    layout = figure.setdefault("layout", {})
    layout["autosize"] = True
    layout.pop("height", None)
    layout.pop("width", None)
    layout["margin"] = {"t": 50, "b": 48, "l": 60, "r": 28}
    return figure


def render_dashboard_html() -> str:
    """Build the full dashboard HTML page from the demo data bundle."""
    d: DemoData = get_demo_data()

    latency = ChartGenerator.latency_timeseries(
        d.timestamps, d.latency_p50, d.latency_p95, d.latency_p99
    )
    throughput = ChartGenerator.throughput_chart(d.timestamps, d.throughput_rps)
    errors = ChartGenerator.error_rate_chart(d.timestamps, d.error_rate)
    drift = ChartGenerator.drift_heatmap(
        [r.feature for r in d.drift.per_feature_scores],
        [r.drift_score for r in d.drift.per_feature_scores],
    )
    distribution = ChartGenerator.prediction_distribution(d.pred_labels, d.pred_counts)

    figs = {
        "latency": _fig_json(_prep(latency)),
        "throughput": _fig_json(_prep(throughput)),
        "errors": _fig_json(_prep(errors)),
        "drift": _fig_json(_prep(drift)),
        "distribution": _fig_json(_prep(distribution)),
    }

    drifted = ", ".join(d.drift.drifted_features) or "none"
    peak_p99 = max(d.latency_p99)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ML Monitoring Dashboard — {d.model_name}</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
  <style>
    :root {{ --bg:#0f172a; --card:#ffffff; --muted:#64748b; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family: system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
            background:#f1f5f9; color:#0f172a; }}
    .banner {{ background:#fef3c7; color:#92400e; padding:10px 20px; font-weight:600;
               border-bottom:1px solid #fde68a; text-align:center; }}
    header {{ background:var(--bg); color:#fff; padding:18px 24px; }}
    header h1 {{ margin:0; font-size:20px; }}
    header .sub {{ color:#94a3b8; font-size:13px; margin-top:4px; }}
    .summary {{ display:flex; gap:16px; flex-wrap:wrap; padding:16px 24px; }}
    .stat {{ background:var(--card); border-radius:10px; padding:12px 18px; min-width:150px;
             box-shadow:0 1px 3px rgba(0,0,0,.08); }}
    .stat .label {{ color:var(--muted); font-size:12px; text-transform:uppercase; }}
    .stat .value {{ font-size:22px; font-weight:700; margin-top:4px; }}
    .grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:20px;
             padding:4px 24px 24px; }}
    .card {{ background:var(--card); border-radius:10px; padding:10px;
             box-shadow:0 1px 3px rgba(0,0,0,.08); }}
    .card.wide {{ grid-column:1 / -1; }}
    /* Definite height so Plotly (responsive) sizes the chart instead of collapsing */
    .plot {{ width:100%; height:320px; }}
    footer {{ color:var(--muted); font-size:12px; text-align:center; padding:16px; }}
    @media (max-width:900px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <div class="banner">⚠ {DEMO_BANNER}</div>
  <header>
    <h1>ML Monitoring Dashboard</h1>
    <div class="sub">Model: <strong>{d.model_name}</strong> · window: last 24h ·
      overall drift score: <strong>{d.drift.overall_drift_score}</strong> ·
      drifted features: <strong>{drifted}</strong></div>
  </header>
  <section class="summary">
    <div class="stat"><div class="label">Predictions (24h)</div>
      <div class="value">{sum(d.pred_counts):,}</div></div>
    <div class="stat"><div class="label">Peak P99 latency</div>
      <div class="value">{peak_p99:.0f} ms</div></div>
    <div class="stat"><div class="label">Peak error rate</div>
      <div class="value">{max(d.error_rate):.1%}</div></div>
    <div class="stat"><div class="label">Drift status</div>
      <div class="value" style="color:{'#dc2626' if d.drift.is_drifted else '#16a34a'}">
        {'DRIFT' if d.drift.is_drifted else 'STABLE'}</div></div>
  </section>
  <main class="grid">
    <div class="card"><div class="plot" id="latency"></div></div>
    <div class="card"><div class="plot" id="throughput"></div></div>
    <div class="card"><div class="plot" id="errors"></div></div>
    <div class="card"><div class="plot" id="distribution"></div></div>
    <div class="card wide"><div class="plot" id="drift"></div></div>
  </main>
  <footer>Synthetic demo data · charts rendered with Plotly · refresh to re-render</footer>
  <script>
    const FIGS = {{
      latency: {figs["latency"]},
      throughput: {figs["throughput"]},
      errors: {figs["errors"]},
      distribution: {figs["distribution"]},
      drift: {figs["drift"]}
    }};
    const cfg = {{ responsive: true, displayModeBar: false }};
    for (const [id, fig] of Object.entries(FIGS)) {{
      Plotly.newPlot(id, fig.data, fig.layout, cfg);
    }}
  </script>
</body>
</html>"""
