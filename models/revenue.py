"""Agent 2 - Revenue prediction.

Target: daily GROSS revenue = everything billed to customers (sum of payments) for all real
orders placed that day, whatever happened to them later. Forecast with a weighted ensemble of
five regressors on lag and calendar features.

Train:    python -m models.revenue          (writes trained_models/revenue.pkl + revenue_forecast)
Predict:  from models import revenue; revenue.predict(horizon_days=30)
"""
from __future__ import annotations

import pandas as pd

from .ensemble import (DATA_END, DATA_START, connect, forecast_summary, load_bundle, save_bundle, save_evaluation,
                       train_daily_forecaster)

AGENT, TARGET = "revenue", "gross"
SERIES_SQL = """
    SELECT purchase_date AS day, SUM(payment_total) AS value
    FROM v_sales
    WHERE purchase_date BETWEEN ? AND ?
    GROUP BY purchase_date
"""


def load_series(con, sql: str = SERIES_SQL) -> pd.Series:
    df = pd.read_sql_query(sql, con, params=(DATA_START, DATA_END), parse_dates=["day"])
    return df.set_index("day")["value"].reindex(pd.date_range(DATA_START, DATA_END, freq="D"), fill_value=0.0).astype(float)


def store_forecast(con, target: str, forecast: pd.Series) -> None:
    con.execute("DELETE FROM revenue_forecast WHERE target = ?", (target,))
    con.executemany("INSERT INTO revenue_forecast VALUES (?, ?, ?)",
                    [(target, str(d.date()), float(v)) for d, v in forecast.items()])
    con.commit()


def train(horizon_days: int = 60) -> dict:
    con = connect()
    try:
        bundle = train_daily_forecaster(load_series(con))
        bundle.update(agent=AGENT, target="daily gross revenue (R$ billed, real orders)")
        save_evaluation(con, AGENT, bundle["evaluation"], bundle["weights"])
        store_forecast(con, TARGET, forecast_summary(bundle, horizon_days)["forecast"])
    finally:
        con.close()
    save_bundle(AGENT, bundle)
    return bundle


def predict(horizon_days: int = 30) -> dict:
    """Forecast the next `horizon_days` after the last day of data, with trend and comparisons."""
    if not 1 <= horizon_days <= 90:
        raise ValueError("horizon_days must be between 1 and 90")
    bundle = load_bundle(AGENT)
    out = forecast_summary(bundle, horizon_days)
    test = bundle["evaluation"].query("split == 'test'").set_index("model")
    out["insights"] = [
        f"Forecast for {out['forecast_start']} to {out['forecast_end']}: R${out['forecast_total']:,.0f} "
        f"({out['growth_vs_previous_period_pct']:+.1f}% vs the previous {horizon_days} days) - revenue is {out['trend']}.",
        f"One-day-ahead test error of the ensemble: MAPE {test.loc['ensemble', 'MAPE']:.1f}% "
        f"(naive same-weekday-last-week: {test.loc['naive_same_weekday_last_week', 'MAPE']:.1f}%).",
        f"Forecasting {bundle['recursive_test']['horizon_days']} days ahead in the test period missed the total by "
        f"{bundle['recursive_test']['total_error_pct']:+.1f}%.",
        "History has one Black Friday (24 Nov 2017) only, so promotional spikes cannot be forecast.",
    ]
    out["evaluation"] = bundle["evaluation"]
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    b = train()
    print("splits:", b["splits"])
    print("weights:", {k: round(v, 3) for k, v in b["weights"].items()})
    print(b["evaluation"].round(3).to_string(index=False))
    print("recursive test:", {k: round(v, 2) for k, v in b["recursive_test"].items()})
    for line in predict(30)["insights"]:
        print(" -", line)
