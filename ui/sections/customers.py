"""C. Customers: actual repeat behaviour and churn-risk scores (model trained partly on synthetic data)."""
from __future__ import annotations

import streamlit as st

from charts import plots
from components import data
from components.format import brl, label, num, pct, tidy_table
from components.widgets import chart, insights, kpi, model_block, section_header, table

TITLE = "Customers"
SYNTHETIC_NOTE = ("Churn scores are model predictions. Only 3.1% of real customers ever placed a second order, so the "
                  "model was trained on real orders plus **synthetic** repeat orders whose return pattern was designed. "
                  "Read the scores as a demonstration of the method, not as evidence about real customers.")
CUSTOMER_COLS = dict(
    customer_unique_id=st.column_config.TextColumn("Customer id"),
    churn_probability=st.column_config.ProgressColumn("Churn probability", min_value=0.0, max_value=1.0, format="%.2f"),
    risk_category=st.column_config.TextColumn("Risk"),
    recency_days=st.column_config.NumberColumn("Days since last order"),
    n_orders=st.column_config.NumberColumn("Orders"),
    total_paid=st.column_config.NumberColumn("Total paid (R$)", format="%.2f"),
)


def lookup() -> None:
    section_header("Look up one customer", "synthetic")
    with st.form("cust-lookup"):
        cid = st.text_input("Customer unique id", placeholder="e.g. 0753264f31df82553453f8f93f65306b", key="cust-id")
        submitted = st.form_submit_button("Score customer")
    if not submitted and not cid:
        return
    cid = (cid or "").strip()
    if not cid:
        st.info("Enter a customer unique id (32 hexadecimal characters).")
        return
    res, err = data.run("churn", quiet=True, customer_unique_id=cid)
    if err:
        if "not found" in err:
            st.warning(f"No scored customer has the id `{cid}`. Check the id and try again.")
        else:
            st.error(err)
        return
    cu = res["customer"]
    c = st.columns(5)
    kpi(c[0], "Churn probability", f"{cu['churn_probability']:.2f}", kind="synthetic")
    kpi(c[1], "Risk category", cu["risk_category"], kind="synthetic",
        note="Predicted to churn" if cu.get("predicted_churn") else "Predicted to return")
    kpi(c[2], "Days since last order", num(cu["recency_days"]), note=f"At {cu['snapshot_date']}")
    kpi(c[3], "Orders", num(cu["n_orders"]))
    kpi(c[4], "Total paid", brl(cu["total_paid"], 2))
    st.caption(res.get("note", "") + " Order counts and spend here are the model's inputs and may include synthetic orders.")


def render() -> None:
    st.title(TITLE)
    ov = data.run("overview")
    top_n = st.slider("Customers listed per table", 5, 100, 20, step=5, key="cust-topn")
    ch = data.run("churn", top_n=top_n)
    if ov is None or ch is None:
        return
    k = ov["kpis"]

    c = st.columns(3)
    kpi(c[0], "Returning customers", f"{num(k['returning_customers'])} of {num(k['total_customers'])}",
        note=f"{pct(k['returning_customer_pct'], 2)} of customers ordered more than once (real orders)")
    kpi(c[1], "Customers scored", num(ch["customers_scored"]), kind="synthetic",
        note=f"Scoring snapshot {ch.get('snapshot_date', '')}")
    kpi(c[2], "Predicted to churn", num(ch["predicted_to_churn"]), kind="synthetic",
        note=f"of {num(ch['customers_scored'])} scored customers")
    st.warning(SYNTHETIC_NOTE)

    left, right = st.columns(2)
    with left:
        section_header("Churn risk distribution", "synthetic")
        o = ch["overview"]
        chart(plots.hbar(o.risk_category + " risk", o.customers, "Customers", color_key="forecast"),
              tidy_table(o, money=("total_paid",)), key="cust-risk")
    with right:
        section_header("What drives the churn score", "synthetic")
        f = ch["factors"]
        chart(plots.hbar(f.description.map(label), f.importance, "Importance", value_format=".3f", color_key="forecast"),
              tidy_table(f[["description", "importance", "effect"]]), key="cust-factors")

    section_header("Customers needing retention attention", "synthetic",
                   "Highest-spending repeat customers the model flags as at risk.")
    table(ch["customers_needing_retention_attention"], **CUSTOMER_COLS)
    section_header("Customers most likely to return", "synthetic")
    table(ch["most_likely_to_return"], **CUSTOMER_COLS)
    insights(ch.get("insights"))

    st.divider()
    lookup()
    st.divider()
    section_header("Model quality")
    model_block(ch, "ROC_AUC", lower_is_better=False, agent_label="Churn")
    cb = ch.get("class_balance") or {}
    if cb:
        st.caption(f"Churn rate in training data {pct(100 * cb['train_churn_rate'])} ({num(cb['train_customers'])} customers); "
                   f"in test data {pct(100 * cb['test_churn_rate'])} ({num(cb['test_customers'])} customers).")
