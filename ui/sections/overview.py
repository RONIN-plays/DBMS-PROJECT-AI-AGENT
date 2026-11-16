"""A. Business overview."""
from __future__ import annotations

import streamlit as st

from charts import plots
from components import data
from components.format import brl, label, md, num, pct, tidy_table
from components.widgets import chart, kpi, section_header

TITLE = "Business overview"


def render() -> None:
    st.title(TITLE)
    ov = data.run("overview")
    if ov is None:
        return
    k = ov["kpis"]
    st.caption(f"Orders from {k['first_order_date']} to {k['last_order_date']} (real Olist orders). "
               f"Only {data.DATA_START} to {data.DATA_END} is complete; the months before and after are tiny extracts.")

    c = st.columns(4)
    kpi(c[0], "Total revenue", brl(k["total_revenue"]), note="All payments billed, every order status",
        help="Sum of payments on all real orders.")
    kpi(c[1], "Net revenue", brl(k["net_revenue"]), note="Excluding cancelled and unavailable orders")
    kpi(c[2], "Total sales (units)", num(k["total_units_sold"]), note=f"Average order value {brl(k['avg_order_value'], 2)}")
    kpi(c[3], "Total orders", num(k["total_orders"]), note=f"{num(k['delivered_orders'])} delivered")
    c = st.columns(4)
    kpi(c[0], "Total customers", num(k["total_customers"]), note="Unique customers")
    kpi(c[1], "Returning customers", num(k["returning_customers"]),
        note=f"{pct(k['returning_customer_pct'], 2)} of customers ordered more than once")
    kpi(c[2], "Average review score", num(k["avg_review_score"], 2), note="Out of 5")
    risk = ov["churn_risk"]
    high = risk["by_category"].set_index("risk_category").customers.get("High")
    kpi(c[3], "High churn risk", num(high), kind="synthetic",
        note=f"of {num(risk['customers_scored'])} customers scored. {risk['note']}")

    st.divider()
    section_header("Monthly revenue", "actual")
    monthly = data.mark_completeness(ov["monthly_revenue"])
    show_edges = st.toggle("Show incomplete edge months", value=False, key="ov-edges",
                           help="Sept 2016 - Dec 2016 and Sept - Oct 2018 hold only a handful of orders in the source extract.")
    fig = plots.monthly_trend(monthly, "payment_total", "Revenue per month", "Revenue (R$)", show_edges=show_edges)
    chart(fig, tidy_table(monthly[["purchase_month", "orders", "customers", "payment_total", "avg_order_value", "completeness"]],
                          money=("payment_total", "avg_order_value"), rename={"payment_total": "Revenue"}), key="ov-monthly")
    st.caption("Hollow markers: January 2017 (data starts 9 Jan) and August 2018 (complete only to 19 Aug). "
               "Edge months are hidden by default because they are an artefact of the extract, not a sales collapse.")

    left, right = st.columns(2)
    with left:
        section_header("Top categories by units sold", "actual")
        tc = ov["top_categories"]
        chart(plots.hbar(tc.category.map(label), tc.units, "Units sold"),
              tidy_table(tc, money=("items_value",), labels=("category",)), key="ov-cat")
    with right:
        section_header("Orders by status", "actual")
        os_ = ov["order_status"]
        chart(plots.hbar(os_.order_status.map(label), os_.orders, "Orders"),
              tidy_table(os_, labels=("order_status",)), key="ov-status")

    left, right = st.columns(2)
    with left:
        section_header("Churn risk overview", "synthetic")
        st.warning("Churn scores are model predictions. The model was trained on real orders plus synthetic repeat "
                   "orders (only 3.1% of real customers ever returned), so treat them as a demonstration.")
        by = risk["by_category"]
        chart(plots.hbar(by.risk_category + " risk", by.customers, "Customers", color_key="forecast"),
              tidy_table(by), key="ov-risk")
    with right:
        st.subheader("Recommendations")
        st.caption("Cross-agent actions from the full analysis (rule-based on the four agents' outputs; "
                   "they mix actual figures and model predictions).")
        with st.spinner("Running all four agents..."):
            recs = data.default_recommendations()
        if recs:
            for rec in recs:
                st.markdown(f"- {md(rec)}")
