"""Agent 1 - Customer churn prediction.

Definition: a customer is CHURNED if they place no order in the 180 days after a cutoff date.
Features describe the customer using only orders placed before the cutoff (and delivery /
review facts only when the delivery happened before the cutoff), so nothing leaks from the future.

Evaluation is out-of-time:
    train snapshot  cutoff 2017-08-23  (label window ends 2018-02-19)   -> split 75/25 into train / validation
    test snapshot   cutoff 2018-02-20  (label window ends 2018-08-19)   -> later, untouched
    scoring         2018-08-20         (every customer, no label yet)   -> churn_predictions table

IMPORTANT: this agent reads v_orders_all, which includes the SYNTHETIC repeat orders
(see olist-analysis/src/synthetic.py). The real data has ~3% repeat customers, too few to
learn from. The return pattern in the synthetic orders was designed, so the scores show how
the system works; they are not evidence about real Olist customers.

Train:    python -m models.churn            (writes trained_models/churn.pkl + churn_predictions)
Predict:  from models import churn; churn.predict()
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .ensemble import (SEED, WeightedEnsemble, classification_metrics, connect, create_prediction_tables, evaluation_table,
                       fit_all, load_bundle, rank_weights, save_bundle, save_evaluation)

AGENT = "churn"
HORIZON_DAYS = 180
TRAIN_CUTOFF, TEST_CUTOFF, SCORE_DATE = "2017-08-23", "2018-02-20", "2018-08-20"
RISK_BANDS = [(0.8, "High"), (0.5, "Medium"), (0.0, "Low")]          # by churn probability
FEATURES = ["recency_days", "tenure_days", "n_orders", "orders_last_90d", "avg_gap_days", "total_paid", "avg_order_value",
            "last_order_value", "avg_items", "last_n_items", "avg_review", "last_review", "late_rate", "last_late",
            "avg_delivery_days", "last_delivery_days", "last_delivery_known", "credit_card_share", "last_credit_card",
            "avg_installments", "voucher_used", "n_categories", "last_category_code", "state_code"]
LABELS = {"recency_days": "days since last order", "tenure_days": "days between first and last order", "n_orders": "number of orders",
          "orders_last_90d": "orders in the last 90 days", "avg_gap_days": "average days between orders", "total_paid": "total amount paid",
          "avg_order_value": "average order value", "last_order_value": "value of last order", "avg_items": "average items per order",
          "last_n_items": "items in last order", "avg_review": "average review score", "last_review": "review score of last order",
          "late_rate": "share of orders delivered late", "last_late": "last order delivered late", "avg_delivery_days": "average delivery time",
          "last_delivery_days": "delivery time of last order", "last_delivery_known": "last order already delivered",
          "credit_card_share": "share of orders paid by credit card", "last_credit_card": "last order paid by credit card",
          "avg_installments": "average instalments", "voucher_used": "has used a voucher", "n_categories": "number of categories bought",
          "last_category_code": "category of last order", "state_code": "customer state"}

ORDERS_SQL = """
    SELECT o.order_id, o.customer_unique_id, o.order_purchase_timestamp AS ts, o.order_delivered_customer_date AS delivered,
           o.order_estimated_delivery_date AS estimated, o.n_items, o.payment_total, o.max_installments, o.voucher_value,
           o.review_score, o.delivery_state AS state, c.category, p.payment_type
    FROM v_orders_all o
    LEFT JOIN (SELECT order_id, category FROM (
                   SELECT i.order_id, pc.category_name_english AS category,
                          ROW_NUMBER() OVER (PARTITION BY i.order_id ORDER BY i.price DESC, i.order_item_id) AS rn
                   FROM order_items i JOIN products pr ON pr.product_id = i.product_id
                   JOIN product_categories pc ON pc.category_id = pr.category_id) WHERE rn = 1) c ON c.order_id = o.order_id
    LEFT JOIN (SELECT order_id, payment_type FROM (
                   SELECT order_id, payment_type,
                          ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY payment_value DESC, payment_sequential) AS rn
                   FROM payments) WHERE rn = 1) p ON p.order_id = o.order_id
"""


def load_orders(con) -> pd.DataFrame:
    return pd.read_sql_query(ORDERS_SQL, con, parse_dates=["ts", "delivered", "estimated"])


def fit_encoders(orders: pd.DataFrame, cutoff: str) -> dict:
    """Category and state codes, learned from orders before the training cutoff only."""
    past = orders[orders.ts < cutoff]
    top = past.category.value_counts().head(20).index
    return dict(category={c: i for i, c in enumerate(top)}, state={s: i for i, s in enumerate(sorted(past.state.unique()))})


def snapshot(orders: pd.DataFrame, cutoff: str, enc: dict, with_label: bool = True) -> pd.DataFrame:
    """One row per customer who ordered before `cutoff`: features as of the cutoff (+ churn label)."""
    cut = pd.Timestamp(cutoff)
    h = orders[orders.ts < cut].sort_values("ts").copy()
    known = h.delivered < cut                                   # delivery outcome already observable at the cutoff
    h["delivery_days"] = ((h.delivered - h.ts).dt.total_seconds() / 86400).where(known)
    h["late"] = (h.delivered > h.estimated).astype(float).where(known)
    h["review"] = h.review_score.where(known)
    h["credit_card"] = (h.payment_type == "credit_card").astype(float)
    h["voucher"] = (h.voucher_value > 0).astype(float)
    h["recent"] = (h.ts >= cut - pd.Timedelta(days=90)).astype(int)
    h["known"] = known.astype(int)

    g = h.groupby("customer_unique_id")
    f = g.agg(n_orders=("order_id", "size"), first=("ts", "min"), last=("ts", "max"), orders_last_90d=("recent", "sum"),
              total_paid=("payment_total", "sum"), avg_order_value=("payment_total", "mean"), avg_items=("n_items", "mean"),
              avg_review=("review", "mean"), late_rate=("late", "mean"), avg_delivery_days=("delivery_days", "mean"),
              credit_card_share=("credit_card", "mean"), avg_installments=("max_installments", "mean"),
              voucher_used=("voucher", "max"), n_categories=("category", "nunique"))
    last = h.drop_duplicates("customer_unique_id", keep="last").set_index("customer_unique_id")
    f["last_order_value"], f["last_n_items"] = last.payment_total, last.n_items
    f["last_review"], f["last_late"], f["last_delivery_days"] = last.review, last.late, last.delivery_days
    f["last_delivery_known"], f["last_credit_card"] = last.known, last.credit_card
    f["last_category_code"] = last.category.map(enc["category"]).fillna(-1)
    f["state_code"] = last.state.map(enc["state"]).fillna(-1)
    f["recency_days"] = (cut - f["last"]).dt.days
    f["tenure_days"] = (f["last"] - f["first"]).dt.days
    f["avg_gap_days"] = (f.tenure_days / (f.n_orders - 1)).where(f.n_orders > 1, -1)
    f = f.drop(columns=["first", "last"])
    if with_label:
        window = orders[(orders.ts >= cut) & (orders.ts < cut + pd.Timedelta(days=HORIZON_DAYS))]
        f["churned"] = (~f.index.isin(window.customer_unique_id)).astype(int)
    return f


def classifiers() -> dict:
    """Five classifiers; missing values (e.g. no delivered order yet) are median-imputed inside each pipeline."""
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "random_forest": make_pipeline(imp(), RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1)),
        "xgboost": make_pipeline(imp(), XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
                                                      colsample_bytree=0.9, eval_metric="logloss", random_state=SEED, n_jobs=4)),
        "gradient_boosting": make_pipeline(imp(), GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=SEED)),
        "decision_tree": make_pipeline(imp(), DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=SEED)),
        "logistic_regression": make_pipeline(imp(), StandardScaler(), LogisticRegression(max_iter=1000)),
    }


def churn_factors(ens: WeightedEnsemble, X: pd.DataFrame, y: pd.Series, n: int = 8000, repeats: int = 3) -> pd.DataFrame:
    """Permutation importance of the ensemble (drop in ROC-AUC when one feature is shuffled) and the direction of its effect."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    Xs, ys = X.iloc[idx].reset_index(drop=True), y.iloc[idx].values
    base_prob = ens.predict(Xs)
    base = roc_auc_score(ys, base_prob)
    rows = []
    for col in X.columns:
        drops = []
        for _ in range(repeats):
            Xp = Xs.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            drops.append(base - roc_auc_score(ys, ens.predict(Xp)))
        corr = pd.Series(base_prob).corr(Xs[col].fillna(Xs[col].median()), method="spearman")
        rows.append(dict(feature=col, description=LABELS[col], importance=float(np.mean(drops)),
                         effect="higher value -> higher churn risk" if corr > 0 else "higher value -> lower churn risk"))
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def risk_category(prob: np.ndarray) -> np.ndarray:
    return np.select([prob >= lo for lo, _ in RISK_BANDS], [name for _, name in RISK_BANDS], default="Low")


def train() -> dict:
    con = connect()
    try:
        orders = load_orders(con)
        enc = fit_encoders(orders, TRAIN_CUTOFF)
        snap_a, snap_b = snapshot(orders, TRAIN_CUTOFF, enc), snapshot(orders, TEST_CUTOFF, enc)
        Xa, ya, Xb, yb = snap_a[FEATURES], snap_a["churned"], snap_b[FEATURES], snap_b["churned"]
        X_tr, X_va, y_tr, y_va = train_test_split(Xa, ya, test_size=0.25, stratify=ya, random_state=SEED)
        templates = classifiers()

        # train -> validate -> rank-based weights (ranked by ROC-AUC, which does not depend on a threshold)
        fitted = fit_all(templates, X_tr, y_tr)
        val_each = {n: m.predict_proba(X_va)[:, 1] for n, m in fitted.items()}
        weights = rank_weights({n: classification_metrics(y_va, p)["ROC_AUC"] for n, p in val_each.items()}, higher_is_better=True)
        val_table = evaluation_table(y_va, val_each, sum(weights[n] * val_each[n] for n in fitted), "validation", classification_metrics)

        # refit on the whole training snapshot; test on the later snapshot
        ens = WeightedEnsemble(fit_all(templates, Xa, ya), weights, "classification")
        test_table = evaluation_table(yb, ens.predict_each(Xb), ens.predict(Xb), "test (later cutoff)", classification_metrics)
        factors = churn_factors(ens, Xb, yb)

        # final model: both labelled snapshots, same weights; score every customer at the end of the data
        final = WeightedEnsemble(fit_all(templates, pd.concat([Xa, Xb]), pd.concat([ya, yb])), weights, "classification")
        current = snapshot(orders, SCORE_DATE, enc, with_label=False)
        prob = final.predict(current[FEATURES])
        scored = pd.DataFrame({"customer_unique_id": current.index, "snapshot_date": SCORE_DATE, "churn_probability": prob.round(4),
                               "predicted_churn": (prob >= 0.5).astype(int), "risk_category": risk_category(prob),
                               "recency_days": current.recency_days.values, "n_orders": current.n_orders.values,
                               "total_paid": current.total_paid.round(2).values})
        con.execute("DROP TABLE IF EXISTS churn_predictions")
        create_prediction_tables(con)
        scored.to_sql("churn_predictions", con, if_exists="append", index=False)
        evaluation = pd.concat([val_table, test_table], ignore_index=True)
        save_evaluation(con, AGENT, evaluation, weights)
    finally:
        con.close()

    bundle = dict(agent=AGENT, target=f"churned = no order within {HORIZON_DAYS} days after the cutoff",
                  ensemble=final, weights=weights, features=FEATURES, encoders=enc, risk_bands=RISK_BANDS, threshold=0.5,
                  evaluation=evaluation, factors=factors, uses_synthetic_orders=True,
                  cutoffs=dict(train=TRAIN_CUTOFF, test=TEST_CUTOFF, scoring=SCORE_DATE, horizon_days=HORIZON_DAYS),
                  class_balance=dict(train_churn_rate=float(ya.mean()), test_churn_rate=float(yb.mean()),
                                     train_customers=len(ya), test_customers=len(yb)))
    save_bundle(AGENT, bundle)
    return bundle


def predict(customer_unique_id: str | None = None, top_n: int = 20) -> dict:
    """Churn scores from the saved ensemble: one customer, or the portfolio overview."""
    b = load_bundle(AGENT)
    con = connect(readonly=True)
    try:
        if customer_unique_id is not None:
            row = pd.read_sql_query("SELECT * FROM churn_predictions WHERE customer_unique_id = ?", con, params=(customer_unique_id,))
            if row.empty:
                raise ValueError(f"customer '{customer_unique_id}' not found in churn_predictions")
            return dict(customer=row.iloc[0].to_dict(), factors=b["factors"].head(8), weights=b["weights"],
                        note="Scores use real and synthetic order history.")
        overview = pd.read_sql_query("""
            SELECT risk_category, COUNT(*) AS customers, ROUND(AVG(churn_probability), 3) AS avg_churn_probability,
                   ROUND(AVG(recency_days)) AS avg_days_since_last_order, ROUND(AVG(n_orders), 2) AS avg_orders,
                   ROUND(SUM(total_paid), 2) AS total_paid
            FROM churn_predictions GROUP BY risk_category
            ORDER BY CASE risk_category WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END""", con)
        # retention targets: valuable repeat customers who are now likely to leave
        attention = pd.read_sql_query("""
            SELECT customer_unique_id, churn_probability, risk_category, recency_days, n_orders, total_paid
            FROM churn_predictions WHERE predicted_churn = 1 AND n_orders >= 2
            ORDER BY total_paid DESC LIMIT ?""", con, params=(top_n,))
        likely_to_return = pd.read_sql_query("""
            SELECT customer_unique_id, churn_probability, risk_category, recency_days, n_orders, total_paid
            FROM churn_predictions ORDER BY churn_probability ASC LIMIT ?""", con, params=(top_n,))
        n_churn = int(pd.read_sql_query("SELECT SUM(predicted_churn) AS n FROM churn_predictions", con).n[0])
    finally:
        con.close()
    total = int(overview.customers.sum())
    high = int(overview.loc[overview.risk_category == "High", "customers"].sum())
    test = b["evaluation"][b["evaluation"].split.str.startswith("test")].set_index("model")
    return dict(
        snapshot_date=b["cutoffs"]["scoring"], customers_scored=total, overview=overview,
        predicted_to_churn=n_churn,
        customers_needing_retention_attention=attention, most_likely_to_return=likely_to_return,
        factors=b["factors"].head(10), weights=b["weights"], evaluation=b["evaluation"], class_balance=b["class_balance"],
        insights=[
            f"{high:,} of {total:,} customers ({100 * high / total:.1f}%) are High risk (churn probability >= 0.8).",
            f"Top churn drivers: {', '.join(b['factors'].description.head(4))}.",
            f"Ensemble on the later test snapshot: ROC-AUC {test.loc['ensemble', 'ROC_AUC']:.3f}, recall {test.loc['ensemble', 'Recall']:.2f}, "
            f"precision {test.loc['ensemble', 'Precision']:.2f}.",
            "Scores are trained on real plus synthetic repeat orders; the return pattern in the synthetic orders was designed, "
            "so treat the scores as a demonstration rather than evidence about real customers.",
        ])


if __name__ == "__main__":
    pd.set_option("display.width", 230, "display.max_columns", 20, "display.max_colwidth", 60)
    b = train()
    print("class balance:", {k: round(v, 3) for k, v in b["class_balance"].items()})
    print("weights:", {k: round(v, 3) for k, v in b["weights"].items()})
    print(b["evaluation"].round(3).to_string(index=False))
    print(b["factors"].head(10).round(4).to_string(index=False))
    res = predict()
    print(res["overview"].to_string(index=False))
    print(res["customers_needing_retention_attention"].head(5).to_string(index=False))
    for line in res["insights"]:
        print(" -", line)
