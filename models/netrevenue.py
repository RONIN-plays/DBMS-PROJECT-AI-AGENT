"""Agent 3 - Net revenue analysis.

Net revenue is CALCULATED from transactions with SQL - no model is involved:

    net revenue = gross billed - cancelled orders - unavailable orders

The dataset has no discounts, returns or refunds, so those deductions are reported as
"not available" instead of being invented. Vouchers and freight are shown as separate
information lines (they are not subtracted: a voucher is a way of paying, and freight is
billed to the customer).

Only the optional FORECAST of future net revenue uses machine learning, with the same
weighted-ensemble pipeline as the revenue agent, trained on the daily net series.

Train:    python -m models.netrevenue       (writes trained_models/netrevenue.pkl)
Analyse:  from models import netrevenue; netrevenue.analyse(); netrevenue.predict(30)
"""
from __future__ import annotations

import pandas as pd

from .ensemble import (DATA_END, DATA_START, connect, forecast_summary, load_bundle, save_bundle, save_evaluation,
                       train_daily_forecaster)
from .revenue import load_series, store_forecast

AGENT, TARGET = "netrevenue", "net"
NOT_AVAILABLE = ["discounts", "returns", "refunds"]
NET_SERIES_SQL = """
    SELECT purchase_date AS day, SUM(payment_total) AS value
    FROM v_sales
    WHERE purchase_date BETWEEN ? AND ? AND order_status NOT IN ('canceled', 'unavailable')
    GROUP BY purchase_date
"""
LEAKAGE_SQL = """
    SELECT {dim} AS segment,
           COUNT(*) AS orders,
           SUM(order_status IN ('canceled', 'unavailable')) AS lost_orders,
           ROUND(SUM(CASE WHEN order_status IN ('canceled', 'unavailable') THEN payment_total ELSE 0 END), 2) AS lost_value,
           ROUND(100.0 * SUM(CASE WHEN order_status IN ('canceled', 'unavailable') THEN payment_total ELSE 0 END)
                 / SUM(payment_total), 2) AS lost_pct_of_billed
    FROM v_sales
    {where}
    GROUP BY 1
    HAVING COUNT(*) >= 200
    ORDER BY lost_pct_of_billed DESC
"""


def analyse(start_month: str | None = None, end_month: str | None = None) -> dict:
    """Exact net-revenue figures for a period ('YYYY-MM' bounds, inclusive; default = all data)."""
    con = connect(readonly=True)
    try:
        monthly = pd.read_sql_query("SELECT * FROM v_net_revenue_monthly ORDER BY purchase_month", con)
        if start_month:
            monthly = monthly[monthly.purchase_month >= start_month]
        if end_month:
            monthly = monthly[monthly.purchase_month <= end_month]
        if monthly.empty:
            raise ValueError("no orders in the requested period")
        where = "WHERE purchase_month BETWEEN '{}' AND '{}'".format(monthly.purchase_month.min(), monthly.purchase_month.max())
        by_state = pd.read_sql_query(LEAKAGE_SQL.format(dim="delivery_state", where=where), con)
        by_payment = pd.read_sql_query(LEAKAGE_SQL.format(
            dim="CASE WHEN voucher_value > 0 THEN 'paid partly by voucher' WHEN max_installments > 1 THEN 'instalments' ELSE 'single payment' END",
            where=where), con)
    finally:
        con.close()

    t = monthly.drop(columns="purchase_month").sum()
    deductions = t.cancelled_value + t.unavailable_value
    monthly = monthly.assign(leakage_pct=(100 * (monthly.cancelled_value + monthly.unavailable_value) / monthly.gross_billed).round(2))
    full = monthly[monthly.gross_billed >= 100_000]            # ignore the near-empty months at the edges of the extract
    worst = (full if len(full) else monthly).sort_values("leakage_pct").iloc[-1]
    return dict(
        period=f"{monthly.purchase_month.min()} to {monthly.purchase_month.max()}",
        gross_revenue=round(float(t.gross_billed), 2),
        deductions=dict(cancelled_orders=round(float(t.cancelled_value), 2), unavailable_orders=round(float(t.unavailable_value), 2),
                        **{name: "not available in the dataset" for name in NOT_AVAILABLE}),
        total_deductions=round(float(deductions), 2),
        net_revenue=round(float(t.net_revenue), 2),
        net_revenue_pct_of_gross=round(float(100 * t.net_revenue / t.gross_billed), 2),
        of_which_delivered=round(float(t.delivered_value), 2),
        of_which_still_in_progress=round(float(t.pending_value), 2),
        information_lines=dict(voucher_payments=round(float(t.voucher_value), 2),
                               freight_on_delivered_orders=round(float(t.freight_delivered), 2)),
        monthly=monthly, leakage_by_state=by_state.head(10), leakage_by_payment=by_payment,
        insights=[
            f"Net revenue is R${t.net_revenue:,.0f}, {100 * t.net_revenue / t.gross_billed:.1f}% of the R${t.gross_billed:,.0f} billed.",
            f"Revenue leakage is R${deductions:,.0f} ({100 * deductions / t.gross_billed:.2f}% of billed): "
            f"R${t.cancelled_value:,.0f} cancelled and R${t.unavailable_value:,.0f} unavailable (out of stock).",
            f"Worst full month: {worst.purchase_month} with {worst.leakage_pct:.2f}% of billed value lost.",
            f"R${t.pending_value:,.0f} is in orders not yet delivered at the end of the data and could still be lost.",
            "Discounts, returns and refunds are not recorded in this dataset, so true net revenue is lower by an unknown amount.",
        ])


def train(horizon_days: int = 60) -> dict:
    con = connect()
    try:
        bundle = train_daily_forecaster(load_series(con, NET_SERIES_SQL))
        bundle.update(agent=AGENT, target="daily net revenue (R$ billed excl. cancelled and unavailable orders)",
                      formula="net = gross billed - cancelled - unavailable", not_available=NOT_AVAILABLE)
        save_evaluation(con, AGENT, bundle["evaluation"], bundle["weights"])
        store_forecast(con, TARGET, forecast_summary(bundle, horizon_days)["forecast"])
    finally:
        con.close()
    save_bundle(AGENT, bundle)
    return bundle


def predict(horizon_days: int = 30) -> dict:
    """Forecast of future net revenue (a prediction - the historical figures come from analyse())."""
    if not 1 <= horizon_days <= 90:
        raise ValueError("horizon_days must be between 1 and 90")
    bundle = load_bundle(AGENT)
    out = forecast_summary(bundle, horizon_days)
    out["evaluation"] = bundle["evaluation"]
    out["insights"] = [
        f"Predicted net revenue {out['forecast_start']} to {out['forecast_end']}: R${out['forecast_total']:,.0f} "
        f"({out['growth_vs_previous_period_pct']:+.1f}% vs the previous {horizon_days} days) - {out['trend']}."]
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 220, "display.max_columns", 20)
    res = analyse()
    for key in ("period", "gross_revenue", "deductions", "total_deductions", "net_revenue", "of_which_delivered",
                "of_which_still_in_progress", "information_lines"):
        print(f"{key}: {res[key]}")
    print(res["monthly"].tail(6).to_string(index=False))
    print(res["leakage_by_state"].head(5).to_string(index=False))
    print(res["leakage_by_payment"].to_string(index=False))
    for line in res["insights"]:
        print(" -", line)
    b = train()
    print("weights:", {k: round(v, 3) for k, v in b["weights"].items()})
    print(b["evaluation"].query("split == 'test'").round(3).to_string(index=False))
    print(" -", predict(30)["insights"][0])
