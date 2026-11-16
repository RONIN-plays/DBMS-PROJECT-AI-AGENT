"""B. Revenue: actual history and the ensemble forecast."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from charts import plots
from components import data
from components.format import brl, md, pct
from components.widgets import chart, insights, kpi, model_block, section_header, table

TITLE = "Revenue"
WINDOWS = {"Last 90 days": 90, "Last 180 days": 180, "All complete history": None}


def history_window(history: pd.Series, choice: str) -> pd.Series:
    days = WINDOWS[choice]
    return history if days is None else history[history.index > history.index.max() - pd.Timedelta(days=days)]


def forecast_table(forecast: pd.Series, column: str = "Forecast (model prediction)") -> pd.DataFrame:
    return pd.DataFrame({"Date": forecast.index.strftime("%Y-%m-%d"), column: forecast.map(brl).values})


def comparison(res: dict, key: str) -> None:
    """Forecast total vs the actual previous period and (when available) the same period last year."""
    h = res["horizon_days"]
    labels, values, kinds = [], [], []
    if res.get("same_period_last_year_total") is not None:
        labels.append("Same period last year")
        values.append(res["same_period_last_year_total"])
        kinds.append("actual")
    labels += [f"Previous {h} days", f"Next {h} days"]
    values += [res["previous_period_total"], res["forecast_total"]]
    kinds += ["actual", "prediction"]
    chart(plots.compare_bars(labels, values, kinds, "Forecast total vs actual comparison periods", "Revenue (R$)"),
          pd.DataFrame({"Period": labels, "Total": [brl(v) for v in values],
                        "Type": ["Actual" if k == "actual" else "Model prediction" for k in kinds]}), key=key)


def render() -> None:
    st.title(TITLE)
    st.caption("Daily revenue = payments on real orders. History is actual; everything from 20 Aug 2018 on is a "
               "model prediction (weighted ensemble of five models).")
    f1, f2 = st.columns(2)
    horizon = f1.slider("Forecast horizon (days)", 7, 90, 30, key="rev-horizon")
    window = f2.radio("History shown", list(WINDOWS), index=1, horizontal=True, key="rev-window")
    res = data.run("revenue", horizon_days=horizon)
    hist = data.daily_history()
    if res is None or hist is None:
        return

    c = st.columns(4)
    ov = data.run("overview")
    kpi(c[0], "Historical revenue", brl(ov["kpis"]["total_revenue"]) if ov else "n/a",
        note=f"All real orders, {ov['kpis']['first_order_date']} to {ov['kpis']['last_order_date']}" if ov else None)
    kpi(c[1], f"Predicted revenue, next {horizon} days", brl(res["forecast_total"]), kind="prediction",
        note=f"{res['forecast_start']} to {res['forecast_end']}")
    kpi(c[2], "Growth vs previous period", pct(res["growth_vs_previous_period_pct"], signed=True), kind="prediction",
        note=f"Trend: {res['trend']}. Previous {horizon} days (actual): {brl(res['previous_period_total'])}")
    sply = res.get("same_period_last_year_total")
    kpi(c[3], "Same period last year", brl(sply) if sply is not None else "Not available",
        note="Actual" if sply is not None else "No data for that period one year earlier")

    fig = plots.history_forecast(history_window(hist, window), res["forecast"], "Daily revenue: actual and forecast",
                                 "Revenue per day (R$)")
    chart(fig, forecast_table(res["forecast"]), key="rev-main")

    left, right = st.columns([3, 2])
    with left:
        comparison(res, key="rev-compare")
    with right:
        insights(res.get("insights"))
        rt = res.get("recursive_test") or {}
        if rt:
            st.caption(md(f"Back-test: forecasting {rt.get('horizon_days')} days ahead in the test period gave "
                       f"{brl(rt.get('forecast_total'))} against an actual {brl(rt.get('actual_total'))} "
                       f"({pct(rt.get('total_error_pct'), signed=True)})."))

    st.divider()
    section_header("Model quality")
    model_block(res, "MAPE", lower_is_better=True, agent_label="Revenue")

    with st.expander("Monthly revenue history (actual)"):
        mh = res["monthly_history"]
        table(pd.DataFrame({"Month": mh.index.strftime("%Y-%m"), "Revenue": mh.map(brl).values}))
        st.caption("August 2018 covers 1-19 August only.")
