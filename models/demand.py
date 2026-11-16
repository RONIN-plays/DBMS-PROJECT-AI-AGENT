"""Agent 4 - Product demand prediction.

Demand is forecast per product CATEGORY per WEEK (units sold), for the 20 largest categories.
Individual products are too sparse to forecast: only 174 sold 50+ units and those have no
sales in 63% of weeks. One model is trained across all categories (lags, trailing means,
calendar, category), combined by the weighted ensemble.

Train:    python -m models.demand           (writes trained_models/demand.pkl + demand_forecast)
Predict:  from models import demand; demand.predict(weeks=4)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .ensemble import (DATA_END, DATA_START, WeightedEnsemble, connect, evaluation_table, fit_all, load_bundle,
                       rank_weights, regression_metrics, regressors, save_bundle, save_evaluation)

AGENT = "demand"
TOP_CATEGORIES = 20
LAGS = [1, 2, 3, 4, 8]
NUMERIC = [f"lag_{k}" for k in LAGS] + ["roll_mean_4", "roll_mean_8", "weekofyear", "month", "trend"]
FEATURES = ["category_code"] + NUMERIC
VAL_WEEKS = TEST_WEEKS = 8


def load_panel(con) -> pd.DataFrame:
    """Weekly units per category (weeks as columns' index), zero-filled, top categories only."""
    df = pd.read_sql_query("SELECT category, week_start, units FROM v_category_weekly_demand WHERE week_start BETWEEN ? AND ?",
                           con, params=(DATA_START, DATA_END), parse_dates=["week_start"])
    top = df[df.category != "unknown"].groupby("category").units.sum().nlargest(TOP_CATEGORIES).index   # 'unknown' is not actionable
    weeks = pd.date_range(DATA_START, DATA_END, freq="W-MON")
    return df[df.category.isin(top)].pivot(index="week_start", columns="category", values="units").reindex(weeks).fillna(0.0)[list(top)]


def make_features(panel: pd.DataFrame, codes: dict[str, int], origin: pd.Timestamp) -> pd.DataFrame:
    """Long table (week, category) with features built only from earlier weeks."""
    parts = []
    for cat in panel.columns:
        s = panel[cat]
        f = pd.DataFrame({f"lag_{k}": s.shift(k) for k in LAGS})
        f["roll_mean_4"] = s.shift(1).rolling(4).mean()
        f["roll_mean_8"] = s.shift(1).rolling(8).mean()
        f["weekofyear"] = s.index.isocalendar().week.astype(int).values
        f["month"] = s.index.month
        f["trend"] = (s.index - origin).days // 7
        f["category_code"] = codes[cat]
        f["category"] = cat
        f["y"] = s
        parts.append(f.rename_axis("week_start").reset_index())
    return pd.concat(parts, ignore_index=True)


def demand_models() -> dict:
    """Five regressors on a log1p target (category volumes differ by an order of magnitude)."""
    base = regressors()
    base["linear_regression"] = make_pipeline(
        ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["category_code"]),
                           ("num", StandardScaler(), NUMERIC)]), Ridge(alpha=1.0))
    return {name: TransformedTargetRegressor(regressor=m, func=np.log1p, inverse_func=np.expm1) for name, m in base.items()}


def recursive_forecast(ens: WeightedEnsemble, panel: pd.DataFrame, weeks: int, codes: dict, origin: pd.Timestamp) -> pd.DataFrame:
    """Forecast `weeks` ahead for every category, feeding predictions back as history."""
    p = panel.copy()
    for _ in range(weeks):
        nxt = p.index[-1] + pd.Timedelta(weeks=1)
        p.loc[nxt] = np.nan
        f = make_features(p.iloc[-12:], codes, origin)
        row = f[f.week_start == nxt]
        p.loc[nxt, row.category.values] = np.clip(ens.predict(row[FEATURES]), 0, None)
    return p.iloc[-weeks:]


def train(weeks_ahead: int = 8) -> dict:
    con = connect()
    try:
        panel = load_panel(con)
        codes = {c: i for i, c in enumerate(panel.columns)}
        origin = panel.index[0]
        data = make_features(panel, codes, origin).dropna()
        test_start = panel.index[-TEST_WEEKS]
        val_start = panel.index[-TEST_WEEKS - VAL_WEEKS]
        tr, va, te = data.week_start < val_start, (data.week_start >= val_start) & (data.week_start < test_start), data.week_start >= test_start
        X, y = data[FEATURES], data["y"]
        templates = demand_models()

        fitted = fit_all(templates, X[tr], y[tr])
        val_each = {n: m.predict(X[va]) for n, m in fitted.items()}
        weights = rank_weights({n: regression_metrics(y[va], p)["RMSE"] for n, p in val_each.items()}, higher_is_better=False)
        val_table = evaluation_table(y[va], val_each, sum(weights[n] * val_each[n] for n in fitted), "validation", regression_metrics)

        fitted = fit_all(templates, X[~te], y[~te])
        ens = WeightedEnsemble(fitted, weights, "regression")
        naive = {"naive_last_week": X.loc[te, "lag_1"].values, "naive_4_week_average": X.loc[te, "roll_mean_4"].values}
        test_table = evaluation_table(y[te], ens.predict_each(X[te]), ens.predict(X[te]), "test", regression_metrics, extra=naive)

        rec = recursive_forecast(ens, panel[panel.index < test_start], TEST_WEEKS, codes, origin)
        actual = panel[panel.index >= test_start]
        recursive = dict(horizon_weeks=TEST_WEEKS, total_error_pct=float((rec.values.sum() / actual.values.sum() - 1) * 100),
                         **regression_metrics(actual.values.ravel(), rec.values.ravel()))

        final = WeightedEnsemble(fit_all(templates, X, y), weights, "regression")
        bundle = dict(agent=AGENT, target="weekly units sold per category (real orders, excl. cancelled/unavailable)",
                      ensemble=final, weights=weights, features=FEATURES, category_codes=codes, origin=origin, history=panel,
                      evaluation=pd.concat([val_table, test_table], ignore_index=True), recursive_test=recursive,
                      splits=dict(validation_start=str(val_start.date()), test_start=str(test_start.date()), test_end=str(panel.index[-1].date())))
        save_evaluation(con, AGENT, bundle["evaluation"], weights)
        fc = recursive_forecast(final, panel, weeks_ahead, codes, origin)
        con.execute("DELETE FROM demand_forecast")
        con.executemany("INSERT INTO demand_forecast VALUES (?, ?, ?)",
                        [(cat, str(week.date()), float(v)) for week, row in fc.iterrows() for cat, v in row.items()])
        con.commit()
    finally:
        con.close()
    save_bundle(AGENT, bundle)
    return bundle


def predict(weeks: int = 4, category: str | None = None) -> dict:
    """Forecast weekly units for the next `weeks` weeks, with demand level, trend and stock advice."""
    if not 1 <= weeks <= 12:
        raise ValueError("weeks must be between 1 and 12")
    b = load_bundle(AGENT)
    panel = b["history"]
    if category is not None and category not in panel.columns:
        raise ValueError(f"'{category}' is not a forecast category. Available: {', '.join(panel.columns)}")
    fc = recursive_forecast(b["ensemble"], panel, weeks, b["category_codes"], b["origin"])
    recent = panel.iloc[-4:].mean()
    s = pd.DataFrame({"avg_weekly_units_last_4_weeks": recent.round(1), "forecast_avg_weekly_units": fc.mean().round(1),
                      "forecast_total_units": fc.sum().round(0)})
    s["change_pct"] = (100 * (s.forecast_avg_weekly_units / s.avg_weekly_units_last_4_weeks - 1)).round(1)
    s["trend"] = np.select([s.change_pct > 10, s.change_pct < -10], ["increasing", "decreasing"], "stable")
    hi, lo = s.forecast_avg_weekly_units.quantile([0.75, 0.25])
    s["demand_level"] = np.select([s.forecast_avg_weekly_units >= hi, s.forecast_avg_weekly_units <= lo], ["high", "low"], "medium")
    advice = {("high", "increasing"): "Raise stock ahead of demand; highest priority for replenishment.",
              ("high", "stable"): "Keep stock at current levels; never allow a stock-out.",
              ("high", "decreasing"): "Still a top seller, but trim the next order to match the decline.",
              ("medium", "increasing"): "Growing: increase stock moderately and watch weekly.",
              ("medium", "stable"): "Reorder at the current rate.",
              ("medium", "decreasing"): "Reduce the next order.",
              ("low", "increasing"): "Small but growing: test with a slightly larger order.",
              ("low", "stable"): "Hold minimal stock.",
              ("low", "decreasing"): "Run down stock; consider promotion or delisting slow items."}
    s["recommendation"] = [advice[(lvl, tr)] for lvl, tr in zip(s.demand_level, s.trend)]
    s = s.sort_values("forecast_avg_weekly_units", ascending=False).rename_axis("category").reset_index()
    weekly = fc.round(1).rename_axis("week_start")
    if category is not None:
        s, weekly = s[s.category == category], weekly[[category]]
    return dict(weeks=weeks, forecast_start=str(fc.index[0].date()), forecast_end=str((fc.index[-1] + pd.Timedelta(days=6)).date()),
                summary=s, weekly_forecast=weekly, history=panel if category is None else panel[[category]],
                high_demand=s.loc[s.demand_level == "high", "category"].tolist(), low_demand=s.loc[s.demand_level == "low", "category"].tolist(),
                increasing=s.loc[s.trend == "increasing", "category"].tolist(), decreasing=s.loc[s.trend == "decreasing", "category"].tolist(),
                weights=b["weights"], evaluation=b["evaluation"], recursive_test=b["recursive_test"],
                note="Forecasts cover the 20 largest named categories by units (about 87% of all units). "
                     "Individual products and smaller categories are too sparse to forecast.")


if __name__ == "__main__":
    pd.set_option("display.width", 230, "display.max_columns", 20, "display.max_colwidth", 70)
    b = train()
    print("splits:", b["splits"])
    print("weights:", {k: round(v, 3) for k, v in b["weights"].items()})
    print(b["evaluation"].round(3).to_string(index=False))
    print("recursive test:", {k: round(v, 2) for k, v in b["recursive_test"].items()})
    print(predict(4)["summary"].drop(columns="recommendation").to_string(index=False))
