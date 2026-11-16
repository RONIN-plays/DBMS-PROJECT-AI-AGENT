"""E. Net revenue: exact calculation from transactions (actual) plus the net revenue forecast (prediction)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from charts import plots
from components import data
from components.format import NOT_AVAILABLE, brl, label, pct, tidy_table
from components.widgets import chart, insights, kpi, model_block, section_header, table
from sections.revenue import comparison, forecast_table

TITLE = "Net revenue"
LEAK_RENAME = {"segment": "Segment", "lost_value": "Lost value", "lost_pct_of_billed": "Lost % of billed"}


def deductions_table(res: dict) -> pd.DataFrame:
    d = res["deductions"]
    rows = [("Gross revenue (billed)", brl(res["gross_revenue"]), "Actual")]
    rows += [("Less: cancelled orders", brl(-d["cancelled_orders"]), "Actual"),
             ("Less: unavailable orders (out of stock)", brl(-d["unavailable_orders"]), "Actual")]
    for name in ("discounts", "returns", "refunds"):
        value = d.get(name)
        shown = NOT_AVAILABLE if value is None or isinstance(value, str) else brl(-value)
        rows.append((f"Less: {name}", shown, "Not recorded" if shown == NOT_AVAILABLE else "Actual"))
    rows += [("Total deductions", brl(-res["total_deductions"]), "Actual"),
             ("Net revenue", brl(res["net_revenue"]), "Actual")]
    return pd.DataFrame(rows, columns=["Line", "Amount", "Status"])


def render() -> None:
    st.title(TITLE)
    st.caption("Net revenue = gross billed - cancelled orders - unavailable orders, calculated exactly from real "
               "transactions. Discounts, returns and refunds are not recorded in this dataset.")
    ov = data.run("overview")
    if ov is None:
        return
    months = ov["monthly_revenue"].purchase_month.tolist()
    f1, f2 = st.columns([3, 2])
    start, end = f1.select_slider("Period (months)", options=months, value=(months[0], months[-1]), key="net-period")
    horizon = f2.slider("Forecast horizon (days)", 7, 90, 30, key="net-horizon")
    params = {"horizon_days": horizon}
    if (start, end) != (months[0], months[-1]):
        params.update(start_month=start, end_month=end)
    res = data.run("net_revenue", **params)
    if res is None:
        return

    st.markdown(f"**Period: {res['period']}**")
    c = st.columns(4)
    kpi(c[0], "Gross revenue", brl(res["gross_revenue"]), note="All payments billed")
    kpi(c[1], "Deductions", brl(res["total_deductions"]), note="Cancelled and unavailable orders only")
    kpi(c[2], "Net revenue", brl(res["net_revenue"]), note=f"{pct(res['net_revenue_pct_of_gross'], 2)} of gross")
    kpi(c[3], "Still in progress", brl(res["of_which_still_in_progress"]),
        note=f"Part of net revenue not yet delivered; delivered: {brl(res['of_which_delivered'])}")

    left, right = st.columns([3, 2])
    with left:
        section_header("From gross to net", "actual")
        table(deductions_table(res))
        st.caption("Discounts, returns and refunds are not available in this dataset - they are not zero, they are "
                   "unknown, so true net revenue is lower by an unknown amount.")
    with right:
        section_header("Information lines", "actual", "Shown for context; not subtracted.")
        info = res["information_lines"]
        table(pd.DataFrame({"Line": ["Voucher payments (a way of paying)", "Freight on delivered orders (billed to customer)"],
                            "Amount": [brl(info["voucher_payments"]), brl(info["freight_on_delivered_orders"])]}))
        insights(res.get("insights"))

    st.divider()
    section_header("Monthly net revenue", "actual")
    monthly = data.mark_completeness(res["monthly"])
    show_edges = st.toggle("Show incomplete edge months", value=False, key="net-edges")
    shown = monthly if show_edges else monthly[monthly.completeness != "edge"]
    if shown.empty:
        st.info("The selected period only contains incomplete edge months. Switch on 'Show incomplete edge months'.")
    else:
        chart(plots.monthly_trend(monthly, "net_revenue", "Net revenue per month", "Net revenue (R$)", show_edges=show_edges),
              tidy_table(monthly.drop(columns=["freight_delivered", "voucher_value"]),
                         money=("gross_billed", "cancelled_value", "unavailable_value", "pending_value", "delivered_value", "net_revenue"),
                         pcts=("leakage_pct",)), key="net-monthly")
        chart(plots.stacked_months(shown, {"cancelled_value": "Cancelled", "unavailable_value": "Unavailable (out of stock)"},
                                   "Revenue lost per month", "Lost value (R$)"), key="net-lost")
        st.caption("Hollow markers mark partial months (Jan 2017 starts 9 Jan; Aug 2018 is complete only to 19 Aug).")

    left, right = st.columns(2)
    with left:
        section_header("Leakage by state", "actual", "States with at least 200 orders, highest share lost first (top 10).")
        s = res["leakage_by_state"]
        chart(plots.hbar(s.segment, s.lost_pct_of_billed, "% of billed value lost", value_format=".2f"),
              tidy_table(s, money=("lost_value",), pcts=("lost_pct_of_billed",), rename=LEAK_RENAME), key="net-state")
    with right:
        section_header("Leakage by payment segment", "actual")
        p = res["leakage_by_payment"]
        chart(plots.hbar(p.segment.map(label), p.lost_pct_of_billed, "% of billed value lost", value_format=".2f"),
              tidy_table(p, money=("lost_value",), pcts=("lost_pct_of_billed",), rename=LEAK_RENAME), key="net-pay")

    st.divider()
    fc = res["forecast"]
    section_header("Net revenue forecast", "prediction",
                   f"Daily net revenue predicted for {fc['forecast_start']} to {fc['forecast_end']}. "
                   "The forecast always continues from the end of the data, whatever period is selected above.")
    c = st.columns(3)
    kpi(c[0], f"Predicted net revenue, next {horizon} days", brl(fc["forecast_total"]), kind="prediction")
    kpi(c[1], "Growth vs previous period", pct(fc["growth_vs_previous_period_pct"], signed=True), kind="prediction",
        note=f"Trend: {fc['trend']}")
    kpi(c[2], f"Previous {horizon} days", brl(fc["previous_period_total"]), note="Actual net revenue")
    hist = data.daily_history(net=True)
    if hist is not None:
        hist = hist[hist.index > hist.index.max() - pd.Timedelta(days=180)]
        chart(plots.history_forecast(hist, fc["forecast"], "Daily net revenue: actual (last 180 days) and forecast",
                                     "Net revenue per day (R$)"),
              forecast_table(fc["forecast"]), key="net-forecast")
    comparison(fc, key="net-compare")
    insights(fc.get("insights"))
    section_header("Forecast model quality")
    model_block(fc, "MAPE", lower_is_better=True, agent_label="Net revenue")
