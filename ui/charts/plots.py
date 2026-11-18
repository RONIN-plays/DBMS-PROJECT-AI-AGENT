"""Plotly chart builders.

Encoding rules used everywhere (validated with the dataviz palette checker, light and dark):
  actual / history   -> blue, solid line
  model prediction   -> orange, dashed line, on a lightly shaded forecast band
  third series       -> aqua (always with a legend and a table view)
  de-emphasised      -> muted grey
One y-axis per chart, 2px lines, thin bars, legend whenever there are two or more series.
Streamlit's own theme supplies background, grid and fonts, so charts follow light and dark mode.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PALETTE = {
    "light": dict(actual="#0B4F9C", forecast="#eb6834", third="#1baf7a", muted="#9AA8BC", band="rgba(235,104,52,0.07)"),
    "dark": dict(actual="#3987e5", forecast="#d95926", third="#199e70", muted="#898781", band="rgba(217,89,38,0.12)"),
}
ACTUAL, FORECAST = "Actual", "Forecast (model prediction)"


def colors() -> dict:
    try:
        mode = st.context.theme.type or "light"
    except Exception:  # noqa: BLE001 - outside a browser session (tests)
        mode = "light"
    return PALETTE.get(mode, PALETTE["light"])


def _money_axis(axis: dict) -> dict:
    return {**axis, "tickprefix": "R$ ", "tickformat": "~s"}


def _layout(fig: go.Figure, title: str | None, x_title: str | None = None, y_title: str | None = None,
            height: int = 360, money_y: bool = False, legend: bool = True) -> go.Figure:
    yaxis = dict(title=y_title, zeroline=False, rangemode="tozero")
    fig.update_layout(
        title=dict(text=title or "", x=0, xanchor="left", font=dict(size=15)),
        height=height, margin=dict(l=8, r=8, t=56 if title else 16, b=8),
        xaxis=dict(title=x_title, showgrid=False),
        yaxis=_money_axis(yaxis) if money_y else yaxis,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title=None),
        hoverlabel=dict(namelength=-1),
    )
    return fig


def history_forecast(history: pd.Series, forecast: pd.Series, title: str, y_title: str, money: bool = True,
                     history_name: str = ACTUAL, height: int = 380) -> go.Figure:
    """Actual history (solid) followed by the model forecast (dashed, shaded band)."""
    c = colors()
    fmt = "R$ %{y:,.0f}" if money else "%{y:,.1f}"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history.values, name=history_name, mode="lines",
                             line=dict(color=c["actual"], width=2), hovertemplate=f"{fmt}<extra>{history_name}</extra>"))
    if len(forecast):
        fig.add_vrect(x0=forecast.index.min(), x1=forecast.index.max(), fillcolor=c["band"], line_width=0, layer="below")
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name=FORECAST, mode="lines+markers" if len(forecast) <= 12 else "lines",
                                 line=dict(color=c["forecast"], width=2, dash="dash"), marker=dict(size=8),
                                 hovertemplate=f"{fmt}<extra>{FORECAST}</extra>"))
        fig.add_annotation(x=forecast.index.min(), y=0, yref="paper", text=f"Forecast from {forecast.index.min():%d %b %Y}",
                           showarrow=False, xanchor="right", yanchor="bottom", xshift=-4,
                           font=dict(size=11, color=c["muted"]))
    _layout(fig, title, None, y_title, height=height, money_y=money)
    fig.update_layout(hovermode="x unified")
    return fig


def monthly_trend(df: pd.DataFrame, value_col: str, title: str, y_title: str, show_edges: bool = False,
                  name: str = ACTUAL, height: int = 360) -> go.Figure:
    """Monthly actuals. `df` needs purchase_month, value_col and a 'completeness' column:
    'complete', 'partial' (hollow marker) or 'edge' (hidden unless show_edges, then grey dots only)."""
    c = colors()
    main = df[df.completeness != "edge"]
    x = pd.to_datetime(main.purchase_month)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=main[value_col], name=name, mode="lines+markers", line=dict(color=c["actual"], width=2),
        marker=dict(size=8, color=[c["actual"] if s == "complete" else "rgba(0,0,0,0)" for s in main.completeness],
                    line=dict(color=c["actual"], width=2)),
        customdata=main.completeness.map({"complete": "", "partial": " (partial month)"}),
        hovertemplate="%{x|%b %Y}%{customdata}<br>R$ %{y:,.0f}<extra></extra>"))
    edges = df[df.completeness == "edge"]
    if show_edges and len(edges):
        fig.add_trace(go.Scatter(x=pd.to_datetime(edges.purchase_month), y=edges[value_col], mode="markers",
                                 name="Incomplete month (not comparable)", marker=dict(size=8, color=c["muted"]),
                                 hovertemplate="%{x|%b %Y} - incomplete extract<br>R$ %{y:,.0f}<extra></extra>"))
    _layout(fig, title, None, y_title, height=height, money_y=True, legend=True)
    fig.update_xaxes(dtick="M2", tickformat="%b\n%Y")
    return fig


def hbar(labels, values, x_title: str, title: str | None = None, emphasis: list[bool] | None = None,
         value_format: str = ",.0f", prefix: str = "", height: int | None = None, color_key: str = "actual") -> go.Figure:
    """Ranked horizontal bars, first item on top. `emphasis` highlights some bars and greys the rest."""
    c = colors()
    labels, values = list(labels), list(values)
    if emphasis is None:
        bar_colors = c[color_key]
    else:
        bar_colors = [c["actual"] if e else c["muted"] for e in emphasis]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker=dict(color=bar_colors, cornerradius=4),
        text=[prefix + format(v, value_format) for v in values],
        textposition="outside", cliponaxis=False,
        hovertemplate=f"%{{y}}<br>{prefix}%{{x:{value_format}}}<extra></extra>"))
    _layout(fig, title, x_title, None, height=height or max(220, 34 * len(labels) + 90), legend=False)
    fig.update_layout(bargap=0.35)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(showgrid=True, showticklabels=False)
    if values:
        fig.update_xaxes(range=[0, max(values) * 1.18])
    return fig


def compare_bars(labels: list[str], values: list[float], kinds: list[str], title: str, y_title: str) -> go.Figure:
    """Totals side by side; actual bars blue, predicted bars orange with a hatch (legend explains both)."""
    c = colors()
    fig = go.Figure()
    for kind, name, color, pattern in (("actual", ACTUAL, c["actual"], ""), ("prediction", FORECAST, c["forecast"], "/")):
        idx = [i for i, k in enumerate(kinds) if k == kind]
        if not idx:
            continue
        fig.add_trace(go.Bar(x=[labels[i] for i in idx], y=[values[i] for i in idx], name=name,
                             marker=dict(color=color, cornerradius=4, pattern=dict(shape=pattern, solidity=0.25)),
                             text=[f"R$ {values[i]:,.0f}" for i in idx], textposition="outside", cliponaxis=False,
                             hovertemplate="%{x}<br>R$ %{y:,.0f}<extra>" + name + "</extra>"))
    _layout(fig, title, None, y_title, height=340, money_y=True)
    fig.update_layout(bargap=0.45)
    fig.update_xaxes(categoryorder="array", categoryarray=labels)
    fig.update_yaxes(range=[0, max(values) * 1.18] if values else None)
    return fig


def stacked_months(df: pd.DataFrame, series: dict[str, str], title: str, y_title: str) -> go.Figure:
    """Stacked monthly bars, one colour per series (max 3), with a legend."""
    c = colors()
    fig = go.Figure()
    for (col, name), color in zip(series.items(), (c["actual"], c["forecast"], c["third"])):
        fig.add_trace(go.Bar(x=pd.to_datetime(df.purchase_month), y=df[col], name=name,
                             marker=dict(color=color, line=dict(width=0)),
                             hovertemplate="%{x|%b %Y}<br>R$ %{y:,.0f}<extra>" + name + "</extra>"))
    _layout(fig, title, None, y_title, height=340, money_y=True)
    fig.update_layout(barmode="stack", bargap=0.3, legend_traceorder="normal")
    fig.update_xaxes(dtick="M2", tickformat="%b\n%Y")
    return fig
