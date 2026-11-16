"""D. Product demand: actual best sellers and the weekly category demand forecast."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from charts import plots
from components import data
from components.format import label, tidy_table
from components.widgets import chart, kpi, model_block, section_header, table

TITLE = "Product demand"
ALL = "All 20 forecast categories"


def name_list(items: list[str]) -> str:
    return ", ".join(label(i) for i in items) if items else "none"


def render() -> None:
    st.title(TITLE)
    ov = data.run("overview")
    if ov is None:
        return

    section_header("Best sellers to date", "actual", "Units sold on real orders, all history.")
    left, right = st.columns(2)
    with left:
        tc = ov["top_categories"]
        chart(plots.hbar(tc.category.map(label), tc.units, "Units sold", title="Top categories"),
              tidy_table(tc, money=("items_value",), labels=("category",)), key="dem-topcat")
    with right:
        st.markdown("**Top products**")
        tp = ov["top_products"]
        table(tidy_table(tp[["category", "units", "items_value", "avg_price", "product_id"]],
                         money=("items_value",), labels=("category",),
                         rename={"items_value": "Sales value", "avg_price": "Avg price (R$)"}),
              height=388, **{"Avg price (R$)": st.column_config.NumberColumn(format="%.2f")})

    st.divider()
    section_header("Demand forecast", "prediction", "Weekly units per category predicted by the weighted ensemble.")
    base = data.run("demand", weeks=4)
    if base is None:
        return
    categories = base["summary"].category.tolist()
    f1, f2 = st.columns(2)
    weeks = f1.slider("Weeks to forecast", 1, 12, 4, key="dem-weeks")
    choice = f2.selectbox("Category", [ALL] + categories, format_func=lambda c: c if c == ALL else label(c), key="dem-cat")
    res = data.run("demand", weeks=weeks, category=None if choice == ALL else choice)
    if res is None:
        return
    st.caption(f"Forecast window {res['forecast_start']} to {res['forecast_end']}. {res.get('note', '')}")

    if choice == ALL:
        c = st.columns(4)
        kpi(c[0], "High-demand categories", str(len(res["high_demand"])), kind="prediction", note=name_list(res["high_demand"]))
        kpi(c[1], "Low-demand categories", str(len(res["low_demand"])), kind="prediction", note=name_list(res["low_demand"]))
        kpi(c[2], "Increasing (> +10%)", str(len(res["increasing"])), kind="prediction", note=name_list(res["increasing"]))
        kpi(c[3], "Decreasing (< -10%)", str(len(res["decreasing"])), kind="prediction", note=name_list(res["decreasing"]))
        chart_cat = st.selectbox("Category to chart", categories, format_func=label, key="dem-chartcat")
    else:
        chart_cat = choice

    hist = res["history"][chart_cat].copy() if chart_cat in res["history"] else pd.Series(dtype=float)
    fc = res["weekly_forecast"][chart_cat].copy() if chart_cat in res["weekly_forecast"] else pd.Series(dtype=float)
    hist.index, fc.index = pd.to_datetime(hist.index), pd.to_datetime(fc.index)
    fig = plots.history_forecast(hist, fc, f"{label(chart_cat)}: weekly units sold and forecast", "Units per week",
                                 money=False)
    chart(fig, pd.DataFrame({"Week starting": fc.index.strftime("%Y-%m-%d"),
                             "Forecast units (model prediction)": fc.round(1).values}), key="dem-main")

    section_header("Trend and stock recommendations", "prediction")
    summary = res["summary"].assign(category=lambda d: d.category.map(label),
                                    trend=lambda d: d.trend.map(label), demand_level=lambda d: d.demand_level.map(label))
    table(summary,
          category=st.column_config.TextColumn("Category"),
          avg_weekly_units_last_4_weeks=st.column_config.NumberColumn("Avg weekly units, last 4 weeks (actual)", format="%.1f"),
          forecast_avg_weekly_units=st.column_config.NumberColumn("Forecast avg weekly units", format="%.1f"),
          forecast_total_units=st.column_config.NumberColumn("Forecast total units", format="%.0f"),
          change_pct=st.column_config.NumberColumn("Change", format="%+.1f%%"),
          trend=st.column_config.TextColumn("Trend"),
          demand_level=st.column_config.TextColumn("Demand level"),
          recommendation=st.column_config.TextColumn("Recommendation", width="large"))

    st.divider()
    section_header("Model quality")
    model_block(res, "WAPE", lower_is_better=True, agent_label="Demand")
