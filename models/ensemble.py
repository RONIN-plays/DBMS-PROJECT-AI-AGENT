"""Shared building blocks for the four agents.

Weighted Ensemble Learning, as used by every predictive agent:
  1. train each algorithm on the training period;
  2. score each one on a VALIDATION period it has not seen;
  3. rank them and give rank-based weights (0.8, 0.5, 0.3, 0.2, 0.1), normalised to sum to 1;
  4. final prediction = sum(weight_i * prediction_i);
  5. compare the ensemble with every single model on a later, untouched TEST period.
Weights come only from validation data, so the test comparison is unbiased.
"""
from __future__ import annotations

import re
import sqlite3
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "database" / "ecommerce.db"
SQL_PATH = ROOT / "database" / "db.sql"
MODEL_DIR = ROOT / "trained_models"

# The extract thins out after 2018-08-20 (orders/day fall from ~300 to <20) and ramps up in
# early January 2017. Models use only the stable window in between (whole weeks, Mon-Sun).
DATA_START = "2017-01-09"
DATA_END = "2018-08-19"

RANK_WEIGHTS = [0.8, 0.5, 0.3, 0.2, 0.1]
SEED = 42


# ---------------------------------------------------------------- database
def connect(readonly: bool = False) -> sqlite3.Connection:
    """Training needs a writable connection (it stores forecasts); prediction paths pass readonly=True."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} not found - run: python database/build_db.py")
    if readonly:
        return sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys = ON")
    create_prediction_tables(con)
    return con


def create_prediction_tables(con: sqlite3.Connection) -> None:
    """Run the 'predictions' section of db.sql (CREATE TABLE IF NOT EXISTS ...)."""
    section = re.split(r"^-- @@SECTION: predictions\s*$", SQL_PATH.read_text(encoding="utf-8"), flags=re.M)[1]
    con.executescript(section)


def save_evaluation(con: sqlite3.Connection, agent: str, table: pd.DataFrame, weights: dict[str, float]) -> None:
    """Store an evaluation table (columns: model, split, <metrics...>) in model_evaluation."""
    long = table.melt(id_vars=["model", "split"], var_name="metric", value_name="value")
    long.insert(0, "agent", agent)
    long["weight"] = long.model.map(weights)
    con.execute("DELETE FROM model_evaluation WHERE agent = ?", (agent,))
    long[["agent", "model", "weight", "split", "metric", "value"]].to_sql("model_evaluation", con, if_exists="append", index=False)
    con.commit()


# ---------------------------------------------------------------- ensemble
def rank_weights(scores: dict[str, float], higher_is_better: bool) -> dict[str, float]:
    """Rank models by validation score and return normalised rank-based weights."""
    order = sorted(scores, key=scores.get, reverse=higher_is_better)
    raw = {name: RANK_WEIGHTS[min(i, len(RANK_WEIGHTS) - 1)] for i, name in enumerate(order)}
    total = sum(raw.values())
    return {name: raw[name] / total for name in order}


class WeightedEnsemble:
    """Fitted estimators combined by fixed, validation-derived weights."""

    def __init__(self, models: dict, weights: dict[str, float], kind: str):
        assert kind in ("regression", "classification")
        assert abs(sum(weights.values()) - 1) < 1e-9, "weights must be normalised"
        self.models, self.weights, self.kind = models, weights, kind

    def predict_each(self, X) -> dict[str, np.ndarray]:
        if self.kind == "classification":
            return {n: m.predict_proba(X)[:, 1] for n, m in self.models.items()}
        return {n: np.asarray(m.predict(X), dtype=float) for n, m in self.models.items()}

    def predict(self, X) -> np.ndarray:
        """Weighted average of predictions (regression) or of class-1 probabilities (classification)."""
        each = self.predict_each(X)
        return sum(self.weights[n] * each[n] for n in self.models)

    predict_proba = predict


def fit_all(templates: dict, X, y) -> dict:
    return {name: clone(model).fit(X, y) for name, model in templates.items()}


# ---------------------------------------------------------------- metrics
def regression_metrics(y_true, y_pred, reference=None) -> dict[str, float]:
    """Error metrics, plus direction metrics when `reference` (the previous period's actual value) is given.

    Direction metrics treat a forecast as a classifier: positive class = "the value goes UP vs the
    previous period". Accuracy / Precision / Recall / F1 then say how well the model calls rises and falls.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    nz = y_true > 0
    out = dict(MAE=mean_absolute_error(y_true, y_pred),
               RMSE=float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
               R2=r2_score(y_true, y_pred),
               MAPE=float(np.mean(np.abs(y_true[nz] - y_pred[nz]) / y_true[nz]) * 100),
               WAPE=float(np.abs(y_true - y_pred).sum() / y_true.sum() * 100))
    if reference is not None:
        ref = np.asarray(reference, dtype=float)
        up_true, up_pred = (y_true > ref).astype(int), (y_pred > ref).astype(int)
        out.update(Accuracy=accuracy_score(up_true, up_pred), Precision=precision_score(up_true, up_pred, zero_division=0),
                   Recall=recall_score(up_true, up_pred, zero_division=0), F1=f1_score(up_true, up_pred, zero_division=0))
    return out


def direction_metrics(reference):
    """regression_metrics with a fixed previous-period reference, for use as an evaluation_table metric_fn."""
    return lambda y_true, y_pred: regression_metrics(y_true, y_pred, reference)


def classification_metrics(y_true, prob, threshold: float = 0.5) -> dict[str, float]:
    pred = (np.asarray(prob) >= threshold).astype(int)
    return dict(Accuracy=accuracy_score(y_true, pred), Precision=precision_score(y_true, pred, zero_division=0),
                Recall=recall_score(y_true, pred), F1=f1_score(y_true, pred),
                ROC_AUC=roc_auc_score(y_true, prob), PR_AUC=average_precision_score(y_true, prob))


def evaluation_table(y_true, each: dict[str, np.ndarray], ensemble_pred, split: str, metric_fn, extra: dict | None = None) -> pd.DataFrame:
    rows = [dict(model=name, split=split, **metric_fn(y_true, pred)) for name, pred in each.items()]
    rows.append(dict(model="ensemble", split=split, **metric_fn(y_true, ensemble_pred)))
    for name, pred in (extra or {}).items():
        rows.append(dict(model=name, split=split, **metric_fn(y_true, pred)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- persistence
def save_bundle(name: str, bundle: dict) -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / f"{name}.pkl"
    joblib.dump(bundle, path, compress=3)
    load_bundle.cache_clear()
    return path


@lru_cache(maxsize=None)
def load_bundle(name: str) -> dict:
    """Load a trained bundle (cached per process; the cache is cleared whenever a bundle is saved)."""
    path = MODEL_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - train it first: python -m models.{name}")
    return joblib.load(path)


# ---------------------------------------------------------------- daily series forecasting
def regressors() -> dict:
    """The five regression algorithms used by the revenue, net-revenue and demand agents."""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor
    return {
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
        "xgboost": XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                                random_state=SEED, n_jobs=4),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, random_state=SEED),
        "decision_tree": DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=SEED),
        "linear_regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    }


DAILY_LAGS = [1, 2, 7, 14, 21, 28]
DAILY_FEATURES = [f"lag_{k}" for k in DAILY_LAGS] + ["roll_mean_7", "roll_mean_28", "roll_std_7", "dayofweek", "dayofmonth", "month", "trend"]


def daily_features(series: pd.Series, origin: pd.Timestamp) -> pd.DataFrame:
    """Features for each day built ONLY from earlier days (lags, trailing means) and the calendar."""
    f = pd.DataFrame(index=series.index)
    for k in DAILY_LAGS:
        f[f"lag_{k}"] = series.shift(k)
    past = series.shift(1)
    f["roll_mean_7"] = past.rolling(7).mean()
    f["roll_mean_28"] = past.rolling(28).mean()
    f["roll_std_7"] = past.rolling(7).std()
    f["dayofweek"] = f.index.dayofweek
    f["dayofmonth"] = f.index.day
    f["month"] = f.index.month
    f["trend"] = (f.index - origin).days
    return f


def recursive_daily_forecast(ensemble: WeightedEnsemble, history: pd.Series, horizon: int, origin: pd.Timestamp) -> pd.Series:
    """Forecast `horizon` days ahead; each prediction is fed back as history for the next day."""
    s = history.copy()
    for _ in range(horizon):
        day = s.index[-1] + pd.Timedelta(days=1)
        s.loc[day] = np.nan
        x = daily_features(s.iloc[-40:], origin).iloc[[-1]][DAILY_FEATURES]
        s.loc[day] = max(float(ensemble.predict(x)[0]), 0.0)
    return s.iloc[-horizon:]


def train_daily_forecaster(series: pd.Series, val_days: int = 56, test_days: int = 56) -> dict:
    """Train / validate / test a weighted ensemble on one daily series and return its bundle."""
    origin = series.index[0]
    feats = daily_features(series, origin)
    data = feats.assign(y=series).dropna()
    X, y = data[DAILY_FEATURES], data["y"]
    test_start = series.index[-1] - pd.Timedelta(days=test_days - 1)
    val_start = test_start - pd.Timedelta(days=val_days)
    tr, va, te = X.index < val_start, (X.index >= val_start) & (X.index < test_start), X.index >= test_start
    templates = regressors()

    # 1-2) fit on train, score on validation -> 3) rank-based weights
    fitted = fit_all(templates, X[tr], y[tr])
    val_each = {n: m.predict(X[va]) for n, m in fitted.items()}
    val_rmse = {n: regression_metrics(y[va], p)["RMSE"] for n, p in val_each.items()}
    weights = rank_weights(val_rmse, higher_is_better=False)
    val_table = evaluation_table(y[va], val_each, sum(weights[n] * val_each[n] for n in fitted), "validation",
                                 direction_metrics(X.loc[va, "lag_1"].values))

    # 5) refit on train+validation, compare on the untouched test period (one day ahead)
    fitted = fit_all(templates, X[~te], y[~te])
    ens = WeightedEnsemble(fitted, weights, "regression")
    naive = {"naive_same_weekday_last_week": X.loc[te, "lag_7"].values}
    test_table = evaluation_table(y[te], ens.predict_each(X[te]), ens.predict(X[te]), "test",
                                  direction_metrics(X.loc[te, "lag_1"].values), extra=naive)

    # honest multi-step check: forecast the whole test period without seeing any of it
    rec = recursive_daily_forecast(ens, series[series.index < test_start], test_days, origin)
    actual = series[series.index >= test_start]
    recursive = dict(horizon_days=test_days, actual_total=float(actual.sum()), forecast_total=float(rec.sum()),
                     total_error_pct=float((rec.sum() / actual.sum() - 1) * 100), **regression_metrics(actual, rec))

    # final model: refit on everything, weights unchanged
    final = WeightedEnsemble(fit_all(templates, X, y), weights, "regression")
    return dict(ensemble=final, weights=weights, features=DAILY_FEATURES, origin=origin, history=series,
                evaluation=pd.concat([val_table, test_table], ignore_index=True), recursive_test=recursive,
                splits=dict(train_end=str((val_start - pd.Timedelta(days=1)).date()), validation_start=str(val_start.date()),
                            test_start=str(test_start.date()), test_end=str(series.index[-1].date())))


def forecast_summary(bundle: dict, horizon: int) -> dict:
    """Daily forecast plus the comparisons the dashboard and chatbot need."""
    hist = bundle["history"]
    fc = recursive_daily_forecast(bundle["ensemble"], hist, horizon, bundle["origin"])
    prev = hist.iloc[-horizon:]
    year_ago = hist.reindex(fc.index - pd.Timedelta(days=364)).dropna()
    monthly = hist.resample("MS").sum()
    return dict(
        forecast=fc.round(2), horizon_days=horizon,
        forecast_start=str(fc.index[0].date()), forecast_end=str(fc.index[-1].date()),
        forecast_total=float(fc.sum()), previous_period_total=float(prev.sum()),
        growth_vs_previous_period_pct=float((fc.sum() / prev.sum() - 1) * 100),
        same_period_last_year_total=float(year_ago.sum()) if len(year_ago) == horizon else None,
        trend="growing" if fc.sum() > prev.sum() * 1.03 else "declining" if fc.sum() < prev.sum() * 0.97 else "stable",
        monthly_history=monthly.round(2), weights=bundle["weights"], recursive_test=bundle["recursive_test"])
