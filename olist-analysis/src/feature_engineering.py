"""Phase 6 (imbalance analysis) and the five analytical datasets.

Every analytical table is built from tables that were first aggregated to the
target grain, with merge(validate=...) guarding against row multiplication, so
revenue is never double-counted.

Leakage rules followed here:
  * the business tables are never resampled;
  * customer/product/seller tables are aggregates over history - pass `snapshot_date`
    to rebuild them from data strictly before a cutoff when they feed a model;
  * the balancing experiment splits first, fits every encoder/imputer on the
    training fold only, resamples the training fold only and scores on the
    untouched test fold.

Run:  python -m src.feature_engineering   (after src.cleaning)
Outputs: data/analytical/*.csv, reports/imbalance_analysis.xlsx
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .cleaning import haversine_km, load_all_processed
from .profiling import ANALYTICAL_DIR, REPORTS_DIR, write_excel
from .validation import order_level_tables

COMPLETE_MONTHS = ("2017-01", "2018-08")   # months with full data; edges are partial
NOT_A_SALE = ["canceled", "unavailable"]


# --------------------------------------------------------------------------
# Analytical datasets
# --------------------------------------------------------------------------
def build_sales(c: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per order. Item, payment and review facts are aggregated to order grain first."""
    agg = order_level_tables(c)
    o = c["orders"][["order_id", "customer_id", "order_status", "order_purchase_timestamp"]]
    s = (o.merge(c["customers"], on="customer_id", how="left", validate="one_to_one")
          .merge(agg["items"], on="order_id", how="left", validate="one_to_one")
          .merge(agg["payments"], on="order_id", how="left", validate="one_to_one")
          .merge(agg["reviews"][["order_id", "review_score"]], on="order_id", how="left", validate="one_to_one"))
    assert len(s) == len(o), "join multiplied order rows"
    ts = s.order_purchase_timestamp
    s["purchase_date"] = ts.dt.date
    s["purchase_year"] = ts.dt.year
    s["purchase_month"] = ts.dt.strftime("%Y-%m")
    s["purchase_quarter"] = ts.dt.to_period("Q").astype(str)
    s["purchase_dayofweek"] = ts.dt.dayofweek
    s["purchase_hour"] = ts.dt.hour
    s["in_complete_month_window"] = s.purchase_month.between(*COMPLETE_MONTHS)
    s["has_items"] = s.n_items.notna()
    for col in ("n_items", "n_products", "n_sellers"):
        s[col] = s[col].fillna(0).astype(int)
    s["order_value"] = s.items_value + s.freight_value
    s["payment_mismatch"] = ((s.payment_total - s.order_value).abs() > 0.05).where(s.payment_total.notna() & s.order_value.notna(), False)
    s["is_delivered"] = s.order_status == "delivered"
    s["is_canceled_or_unavailable"] = s.order_status.isin(NOT_A_SALE)
    s["review_score"] = s.review_score.astype("Int64")
    lead = ["order_id", "customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_city", "customer_state",
            "order_status", "is_delivered", "is_canceled_or_unavailable", "order_purchase_timestamp", "purchase_date",
            "purchase_year", "purchase_quarter", "purchase_month", "purchase_dayofweek", "purchase_hour", "in_complete_month_window"]
    return s[lead + [x for x in s.columns if x not in lead]]


def build_customers(c: dict[str, pd.DataFrame], sales: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    """One row per customer_unique_id with raw RFM inputs.
    Only orders purchased before `snapshot_date` are used (default: day after the last purchase)."""
    snapshot = pd.Timestamp(snapshot_date) if snapshot_date else sales.order_purchase_timestamp.max().normalize() + pd.Timedelta(days=1)
    s = sales[sales.order_purchase_timestamp < snapshot].sort_values("order_purchase_timestamp")
    s = s.assign(delivered_payment=s.payment_total.where(s.is_delivered))
    g = s.groupby("customer_unique_id")
    cu = g.agg(n_orders=("order_id", "size"), n_delivered_orders=("is_delivered", "sum"),
               first_purchase=("order_purchase_timestamp", "min"), last_purchase=("order_purchase_timestamp", "max"),
               total_items=("n_items", "sum"), total_payment_value=("payment_total", "sum"),
               delivered_payment_value=("delivered_payment", "sum"),
               avg_review_score=("review_score", lambda x: x.astype("float").mean()), n_reviews=("review_score", "count"),
               customer_zip_code_prefix=("customer_zip_code_prefix", "last"), customer_city=("customer_city", "last"),
               customer_state=("customer_state", "last"))
    pay_mode = (s.dropna(subset=["main_payment_type"]).groupby(["customer_unique_id", "main_payment_type"]).size()
                 .rename("n").reset_index().sort_values(["customer_unique_id", "n"], ascending=[True, False])
                 .drop_duplicates("customer_unique_id").set_index("customer_unique_id").main_payment_type)
    cu["preferred_payment_type"] = pay_mode
    cu["snapshot_date"] = snapshot.date()
    cu["recency_days"] = (snapshot - cu.last_purchase).dt.days
    cu["tenure_days"] = (cu.last_purchase - cu.first_purchase).dt.days
    cu["frequency"] = cu.n_orders
    cu["monetary"] = cu.total_payment_value
    cu["avg_order_value"] = (cu.total_payment_value / cu.n_orders).round(2)
    cu["is_repeat_customer"] = cu.n_orders > 1
    cu["avg_days_between_orders"] = (cu.tenure_days / (cu.n_orders - 1)).where(cu.is_repeat_customer).round(1)
    cu["avg_review_score"] = cu.avg_review_score.round(3)
    geo = c["geo_zip"].set_index("zip_code_prefix")
    cu["customer_lat"] = cu.customer_zip_code_prefix.map(geo.lat)
    cu["customer_lng"] = cu.customer_zip_code_prefix.map(geo.lng)
    return cu.reset_index()


def _order_reviews(c: dict[str, pd.DataFrame]) -> pd.Series:
    r = c["reviews"]
    return r[r.is_latest_review_for_order].set_index("order_id").review_score


def build_products(c: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per product. Review score is an approximation: reviews belong to orders, not products."""
    it = c["order_items"].merge(c["orders"][["order_id", "order_status", "order_purchase_timestamp"]], on="order_id", validate="many_to_one")
    g = it.groupby("product_id")
    agg = g.agg(items_sold=("order_item_id", "size"), n_orders=("order_id", "nunique"), n_sellers=("seller_id", "nunique"),
                revenue=("price", "sum"), freight_total=("freight_value", "sum"), avg_price=("price", "mean"),
                min_price=("price", "min"), max_price=("price", "max"),
                first_sale=("order_purchase_timestamp", "min"), last_sale=("order_purchase_timestamp", "max"))
    agg["items_sold_delivered"] = it[it.order_status == "delivered"].groupby("product_id").size()
    pairs = it[["order_id", "product_id"]].drop_duplicates()
    pairs["score"] = pairs.order_id.map(_order_reviews(c))
    agg["avg_review_score_approx"] = pairs.groupby("product_id").score.mean().round(3)
    agg["n_reviewed_orders"] = pairs.groupby("product_id").score.count()
    p = c["products"].merge(agg.reset_index(), on="product_id", how="left", validate="one_to_one")
    p["product_volume_cm3"] = p.product_length_cm * p.product_height_cm * p.product_width_cm
    for col in ("items_sold", "n_orders", "n_sellers", "items_sold_delivered", "n_reviewed_orders"):
        p[col] = p[col].fillna(0).astype(int)
    p[["revenue", "freight_total"]] = p[["revenue", "freight_total"]].fillna(0).round(2)
    p["avg_price"] = p.avg_price.round(2)
    return p


def build_sellers(c: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per seller: volume, GMV, assortment, punctuality and (approximate) satisfaction."""
    o = c["orders"]
    it = (c["order_items"].merge(o[["order_id", "order_status", "order_purchase_timestamp", "order_delivered_carrier_date",
                                    "order_delivered_customer_date", "order_estimated_delivery_date"]], on="order_id", validate="many_to_one")
          .merge(c["products"][["product_id", "product_category_name_english"]], on="product_id", validate="many_to_one"))
    g = it.groupby("seller_id")
    agg = g.agg(items_sold=("order_item_id", "size"), n_orders=("order_id", "nunique"), n_products=("product_id", "nunique"),
                n_categories=("product_category_name_english", "nunique"), revenue=("price", "sum"),
                freight_total=("freight_value", "sum"), avg_price=("price", "mean"),
                first_sale=("order_purchase_timestamp", "min"), last_sale=("order_purchase_timestamp", "max"),
                active_months=("order_purchase_timestamp", lambda x: x.dt.to_period("M").nunique()))
    # one row per (order, seller) for punctuality and reviews, so multi-item orders are not over-weighted
    os_ = it.groupby(["order_id", "seller_id"]).agg(limit=("shipping_limit_date", "max"), carrier=("order_delivered_carrier_date", "first"),
                                                    delivered=("order_delivered_customer_date", "first"),
                                                    estimated=("order_estimated_delivery_date", "first")).reset_index()
    os_["late_shipment"] = (os_.carrier > os_.limit).where(os_.carrier.notna())
    os_["late_delivery"] = (os_.delivered > os_.estimated).where(os_.delivered.notna())
    os_["score"] = os_.order_id.map(_order_reviews(c))
    sg = os_.groupby("seller_id")
    agg["late_shipment_rate"] = sg.late_shipment.apply(lambda x: x.dropna().astype(float).mean()).round(4)
    agg["late_delivery_rate"] = sg.late_delivery.apply(lambda x: x.dropna().astype(float).mean()).round(4)
    agg["avg_review_score_approx"] = sg.score.mean().round(3)
    agg["n_reviewed_orders"] = sg.score.count()
    s = c["sellers"].merge(agg.reset_index(), on="seller_id", how="left", validate="one_to_one")
    geo = c["geo_zip"].set_index("zip_code_prefix")
    s["seller_lat"] = s.seller_zip_code_prefix.map(geo.lat)
    s["seller_lng"] = s.seller_zip_code_prefix.map(geo.lng)
    s[["revenue", "freight_total", "avg_price"]] = s[["revenue", "freight_total", "avg_price"]].round(2)
    return s


def build_delivery(c: dict[str, pd.DataFrame], sales: pd.DataFrame) -> pd.DataFrame:
    """One row per order: stage durations, delay against the promise, route and load.
    Durations exist only where both timestamps exist; nothing is imputed."""
    o = c["orders"]
    day = 86400.0
    d = o[["order_id", "order_status", "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
           "order_delivered_customer_date", "order_estimated_delivery_date", "flag_status_date_conflict", "flag_date_sequence_anomaly"]].copy()
    p, a, ca, cu, e = (d[x] for x in d.columns[2:7])
    d["is_delivered"] = d.order_status == "delivered"
    d["approval_hours"] = ((a - p).dt.total_seconds() / 3600).round(2)
    d["handling_days"] = ((ca - p).dt.total_seconds() / day).round(2)
    d["carrier_days"] = ((cu - ca).dt.total_seconds() / day).round(2)
    d["delivery_days"] = ((cu - p).dt.total_seconds() / day).round(2)
    d["estimated_days"] = ((e - p).dt.total_seconds() / day).round(2)
    d["delay_days"] = ((cu - e).dt.total_seconds() / day).round(2)
    d["is_late"] = (d.delay_days > 0).astype("boolean").where(cu.notna())

    it = c["order_items"].merge(c["products"][["product_id", "product_weight_g", "product_category_name_english"]], on="product_id", validate="many_to_one")
    main = it.sort_values(["order_id", "price", "order_item_id"], ascending=[True, False, True]).drop_duplicates("order_id") \
             .set_index("order_id")[["seller_id", "product_category_name_english"]]
    main.columns = ["main_seller_id", "main_category"]
    d = d.merge(main.reset_index(), on="order_id", how="left", validate="one_to_one")
    d["total_weight_g"] = d.order_id.map(it.groupby("order_id").product_weight_g.sum())
    d = d.merge(sales[["order_id", "customer_zip_code_prefix", "customer_state", "n_items", "n_sellers", "items_value",
                       "freight_value", "review_score"]], on="order_id", how="left", validate="one_to_one")
    se = c["sellers"].set_index("seller_id")
    d["seller_state"] = d.main_seller_id.map(se.seller_state)
    d["same_state"] = (d.seller_state == d.customer_state).astype("boolean").where(d.seller_state.notna())
    geo = c["geo_zip"].set_index("zip_code_prefix")
    szip = d.main_seller_id.map(se.seller_zip_code_prefix)
    d["distance_km"] = haversine_km(d.customer_zip_code_prefix.map(geo.lat).astype(float), d.customer_zip_code_prefix.map(geo.lng).astype(float),
                                    szip.map(geo.lat).astype(float), szip.map(geo.lng).astype(float)).round(1)
    return d.drop(columns="customer_zip_code_prefix")


def analytical_dictionary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Classify every analytical column: measured vs derived vs approximate, and when it becomes known."""
    outcome = ("review", "delay", "is_late", "delivered", "carrier", "handling", "delivery_days", "order_status", "is_canceled", "flag_")
    rows = []
    for name, df in tables.items():
        for col in df.columns:
            if "approx" in col:
                kind = "approximation - review belongs to the whole order"
            elif col in ("order_id", "customer_id", "customer_unique_id", "product_id", "seller_id") or col.startswith(("order_", "customer_", "seller_", "product_")) \
                    and not col.endswith(("_lat", "_lng", "_cm3")):
                kind = "measured (source column)"
            else:
                kind = "derived (deterministic from measured columns)"
            if name in ("customers", "products", "sellers") and kind != "measured (source column)":
                when = "aggregate over history up to the snapshot - rebuild with a cutoff before using as a model feature"
            elif any(k in col for k in outcome) and col not in ("order_purchase_timestamp", "order_estimated_delivery_date", "order_id"):
                when = "after fulfilment / review - outcome, not a purchase-time feature"
            elif "payment" in col or "installments" in col or col == "approval_hours":
                when = "at payment approval"
            else:
                when = "at purchase"
            rows.append(dict(table=name, column=col, dtype=str(df[col].dtype), kind=kind, known_when=when))
    notes = pd.DataFrame([
        dict(table="(all)", column="revenue / items_value", dtype="", kind="measured GMV (sum of item prices)",
             known_when="Olist's own revenue (commission), cost and margin are NOT in the data"),
        dict(table="(all)", column="customer lifetime value, churn", dtype="", kind="NOT provided - must be estimated",
             known_when="no CLV or churn label exists; any such figure is a model output, not a measurement")])
    return pd.concat([pd.DataFrame(rows), notes], ignore_index=True)


# --------------------------------------------------------------------------
# Phase 6: imbalance analysis
# --------------------------------------------------------------------------
IMBALANCE_NOTES = {
    "order_status": ("natural business imbalance",
                     "97% delivered reflects a healthy fulfilment funnel; the other statuses are in-flight or failed orders.",
                     "Do not balance. For a cancellation model merge to binary (canceled+unavailable vs rest) and use class weights with a stratified/time split; 5 'created' and 2 'approved' rows are too few to model."),
    "customer_type": ("natural - genuine retention challenge",
                      "Only ~3% of customers ever return; this is the business finding, not a data defect.",
                      "Do not balance. For churn/repeat models use class weights and PR-AUC; build labels with a time cutoff."),
    "review_score": ("natural, moderately skewed (J-shaped)",
                     "Most customers are satisfied; 1-star is the second largest class, 2-star the rarest.",
                     "For satisfaction models prefer binary low (1-2) vs rest with class weights; SMOTENC on training data only if recall is insufficient."),
    "payment_type": ("natural business imbalance",
                     "Credit card dominates Brazilian e-commerce; debit card is rare; 3 'not_defined' rows are noise.",
                     "Do not balance. As a feature, fold 'not_defined' into 'other'."),
    "product_category": ("natural long-tail demand",
                         "A handful of categories carry most sales; many categories have well under 1% of items.",
                         "Do not balance. For modelling, group categories below ~0.5% share into 'other' or use target/frequency encoding fitted on training data."),
    "seller_volume": ("natural marketplace concentration",
                      "Most sellers make a few sales while a small group makes thousands (see concentration sheet).",
                      "Do not balance. Use per-seller minimum-volume thresholds before ranking sellers on rates."),
    "customer_state": ("natural - follows population and logistics",
                       "Sao Paulo alone holds ~42% of customers; northern states are very small.",
                       "Do not balance. Aggregate to region for small states; stratify by state when sampling."),
}


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def imbalance_distributions(c: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    it = c["order_items"]
    cat = it.product_id.map(c["products"].set_index("product_id").product_category_name_english)
    orders_per_customer = c["customers"].groupby("customer_unique_id").size()
    orders_per_seller = it.groupby("seller_id").order_id.nunique()
    bins = pd.cut(orders_per_seller, [0, 1, 10, 50, 200, 1000, np.inf],
                  labels=["1 order", "2-10", "11-50", "51-200", "201-1000", ">1000"])
    return {
        "order_status": c["orders"].order_status.value_counts(),
        "customer_type": orders_per_customer.gt(1).map({True: "repeat customer", False: "one-time customer"}).value_counts(),
        "review_score": c["reviews"].review_score.map(lambda v: f"{v}-star").value_counts(),
        "payment_type": c["payments"].payment_type.value_counts(),
        "product_category": cat.value_counts(),
        "seller_volume": bins.value_counts(),
        "customer_state": c["customers"].customer_state.value_counts(),
    }


def imbalance_tables(dist: dict[str, pd.Series], rare_pct: float = 1.0) -> dict[str, pd.DataFrame]:
    units = {"order_status": "orders", "customer_type": "unique customers", "review_score": "reviews", "payment_type": "payment records",
             "product_category": "items sold", "seller_volume": "sellers (by number of orders)", "customer_state": "orders by customer state"}
    classes, summary = [], []
    for var, vc in dist.items():
        vc = vc[vc > 0].sort_values(ascending=False)
        pct = 100 * vc / vc.sum()
        df = pd.DataFrame({"variable": var, "class": vc.index.astype(str), "count": vc.values, "pct": pct.round(3).values,
                           "ratio_majority_to_class": (vc.iloc[0] / vc).round(1).values,
                           "role": ["majority"] + [""] * (len(vc) - 2) + ["minority"],
                           f"underrepresented_lt_{rare_pct:g}pct": (pct < rare_pct).values})
        classes.append(df)
        prob = (vc / vc.sum()).values
        nature, reading, advice = IMBALANCE_NOTES[var]
        summary.append(dict(
            variable=var, unit=units[var], n_classes=len(vc), total=int(vc.sum()),
            majority_class=str(vc.index[0]), majority_pct=round(pct.iloc[0], 2),
            minority_class=str(vc.index[-1]), minority_pct=round(pct.iloc[-1], 4),
            imbalance_ratio=round(vc.iloc[0] / vc.iloc[-1], 1),
            classes_below_1pct=int((pct < rare_pct).sum()),
            normalised_entropy=round(float(-(prob * np.log(prob)).sum() / np.log(len(vc))), 3),
            assessment=nature, interpretation=reading, ml_recommendation=advice))
    return {"summary": pd.DataFrame(summary), "class_distribution": pd.concat(classes, ignore_index=True)}


def concentration(c: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """How unevenly sales are spread across sellers, products, categories and customers."""
    it = c["order_items"]
    cat = it.product_id.map(c["products"].set_index("product_id").product_category_name_english)
    series = {"sellers (GMV)": it.groupby("seller_id").price.sum(), "sellers (orders)": it.groupby("seller_id").order_id.nunique(),
              "products (items sold)": it.groupby("product_id").size(), "categories (items sold)": it.groupby(cat).size(),
              "customers (orders)": c["customers"].groupby("customer_unique_id").size(),
              "customer states (orders)": c["customers"].groupby("customer_state").size()}
    rows = []
    for name, s in series.items():
        v = np.sort(s.values)[::-1]
        share = lambda q: round(100 * v[:max(int(np.ceil(len(v) * q)), 1)].sum() / v.sum(), 1)
        rows.append(dict(entity=name, n_entities=len(v), gini=round(gini(v), 3), top_1pct_share=share(.01), top_10pct_share=share(.10),
                         top_20pct_share=share(.20), median=float(np.median(v)), max=float(v[0]),
                         entities_with_single_unit=int((v == 1).sum())))
    return pd.DataFrame(rows)


def imbalance_chart(dist: dict[str, pd.Series]) -> io.BytesIO:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, blue, surface = "#0b0b0b", "#52514e", "#2a78d6", "#fcfcfb"
    titles = {"order_status": "Order status (orders)", "customer_type": "Customer type (unique customers)", "review_score": "Review score (reviews)",
              "payment_type": "Payment method (payment records)", "product_category": "Product category (items sold) - top 12",
              "seller_volume": "Sellers by number of orders", "customer_state": "Customer state (orders) - top 12"}
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5), facecolor=surface)
    for ax, (var, vc) in zip(axes.ravel(), dist.items()):
        vc = vc[vc > 0]
        if var == "review_score":
            vc = vc.sort_index()
        elif var == "seller_volume":
            vc = vc.sort_index()
        else:
            vc = vc.sort_values(ascending=False)
            if len(vc) > 12:
                rest = vc.iloc[12:]
                vc = pd.concat([vc.iloc[:12], pd.Series({f"other ({len(rest)})": rest.sum()})])
        pct = 100 * vc / vc.sum()
        y = np.arange(len(vc))[::-1]
        ax.barh(y, pct.values, height=0.62, color=blue)
        for yi, p_, n in zip(y, pct.values, vc.values):
            ax.text(p_ + pct.max() * 0.015, yi, f"{p_:.1f}%" if p_ >= 0.1 else f"{p_:.3f}%  (n={n})", va="center", fontsize=8, color=muted)
        ax.set_yticks(y, [str(v)[:26] for v in vc.index], fontsize=8.5, color=ink)
        ax.set_xlim(0, pct.max() * 1.28)
        ax.set_title(titles[var], fontsize=10.5, color=ink, loc="left")
        ax.set_facecolor(surface)
        ax.tick_params(axis="x", colors=muted, labelsize=8)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#e6e5e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c2")
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].text(0, 0.95, "Reading the charts", fontsize=10.5, color=ink, va="top")
    axes.ravel()[-1].text(0, 0.84, "Bars show each class as a share of its\nvariable's total. Every distribution here is a\nnatural business pattern, so the business\ndata is never resampled.\n\n"
                                   "Balancing is evaluated only inside a model\ntraining fold (see balancing_* sheets).", fontsize=9, color=muted, va="top", linespacing=1.5)
    fig.suptitle("Class distributions of the key categorical variables (share of total, %)", fontsize=13, color=ink, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=surface)
    plt.close(fig)
    return buf


BALANCING_GUIDANCE = pd.DataFrame([
    ("Stratified sampling", "Always - for the train/test split and for cross-validation folds", "Keeps the true class ratio in every fold; no data altered", "Use by default (or a time-based split for forecasting / churn)"),
    ("Class-weight adjustment", "First choice for imbalanced targets (low review, cancellation, repeat purchase)", "No synthetic rows, no information discarded, works with categorical features", "Recommended"),
    ("Random undersampling", "Very large majority class and training cost matters", "Discards real majority data; higher variance", "Acceptable on training fold only"),
    ("Random oversampling", "Small data sets, tree models", "Duplicates minority rows -> risk of over-fitting", "Acceptable on training fold only"),
    ("SMOTE", "All-numeric feature spaces", "Interpolates between neighbours; meaningless on category codes", "Not suitable as-is: Olist features are mixed"),
    ("SMOTENC", "Mixed numeric + categorical features", "Slower; synthetic rows can blur class boundaries", "The appropriate SMOTE variant here - training fold only"),
    ("ADASYN", "All-numeric spaces with hard-to-learn minority regions", "No categorical variant; amplifies noise near the boundary", "Only on one-hot/numeric encodings; generally not preferred"),
], columns=["technique", "when_appropriate", "caveats", "verdict_for_olist"])


def evaluate_balancing(sales: pd.DataFrame, delivery: pd.DataFrame, random_state: int = 42) -> dict[str, pd.DataFrame]:
    """Leak-free comparison of balancing techniques on an illustrative target:
    low review (score <= 2) predicted from purchase-time features only."""
    from imblearn.over_sampling import ADASYN, SMOTENC, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    from sklearn.model_selection import train_test_split

    num = ["items_value", "freight_value", "n_items", "n_sellers", "max_installments", "estimated_days", "total_weight_g",
           "distance_km", "purchase_dayofweek", "purchase_hour"]
    cat = ["main_payment_type", "customer_state", "seller_state", "main_category"]
    d = sales[["order_id", "review_score", "has_items", "max_installments", "main_payment_type", "purchase_dayofweek", "purchase_hour"]].merge(
        delivery[["order_id", "items_value", "freight_value", "n_items", "n_sellers", "estimated_days", "total_weight_g", "distance_km",
                  "customer_state", "seller_state", "main_category"]], on="order_id", validate="one_to_one")
    d = d[d.review_score.notna() & d.has_items & d.main_payment_type.notna()]
    y = (d.review_score <= 2).astype(int).values
    X = d[num + cat]

    # 1) split FIRST (stratified); everything below is fitted on the training fold only
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)

    med = X_tr[num].median()
    levels = {col: X_tr[col].value_counts().loc[lambda s: s >= 50].index for col in cat}   # rare levels -> 'other'

    def encode(frame: pd.DataFrame, onehot: bool) -> pd.DataFrame:
        out = frame[num].astype(float).fillna(med)
        for col in cat:
            v = frame[col].where(frame[col].isin(levels[col]), "other").astype(str)
            if onehot:
                dummies = pd.get_dummies(pd.Categorical(v, categories=list(levels[col]) + ["other"]), prefix=col, dtype=float)
                dummies.index = frame.index
                out = pd.concat([out, dummies], axis=1)
            else:
                out[col] = pd.Categorical(v, categories=list(levels[col]) + ["other"]).codes
        return out

    Xtr_o, Xte_o = encode(X_tr, False), encode(X_te, False)       # ordinal codes (native categorical support)
    Xtr_h, Xte_h = encode(X_tr, True), encode(X_te, True)         # one-hot, for ADASYN
    cat_idx = [Xtr_o.columns.get_loc(col) for col in cat]

    samplers = {
        "No balancing (baseline)": (None, False, None),
        "Class weights": (None, False, "balanced"),
        "Random oversampling": (RandomOverSampler(random_state=random_state), False, None),
        "Random undersampling": (RandomUnderSampler(random_state=random_state), False, None),
        "SMOTENC (categorical-aware)": (SMOTENC(categorical_features=cat_idx, random_state=random_state), False, None),
        "ADASYN (on one-hot encoding)": (ADASYN(random_state=random_state), True, None),
    }
    counts, metrics = [], []
    for label, (sampler, onehot, weight) in samplers.items():
        Xa, Xb = (Xtr_h, Xte_h) if onehot else (Xtr_o, Xte_o)
        Xs, ys = (Xa, y_tr) if sampler is None else sampler.fit_resample(Xa, y_tr)      # 2) resample TRAIN only
        model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=random_state, class_weight=weight,
                                               categorical_features=None if onehot else cat_idx)
        model.fit(Xs, ys)
        prob = model.predict_proba(Xb)[:, 1]                                              # 3) score on untouched TEST
        pred = (prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
        counts.append(dict(technique=label, train_majority_before=int((y_tr == 0).sum()), train_minority_before=int((y_tr == 1).sum()),
                           train_minority_pct_before=round(100 * y_tr.mean(), 2),
                           train_majority_after=int((ys == 0).sum()), train_minority_after=int((ys == 1).sum()),
                           train_minority_pct_after=round(100 * np.mean(ys), 2),
                           test_majority=int((y_te == 0).sum()), test_minority=int((y_te == 1).sum()),
                           test_minority_pct=round(100 * y_te.mean(), 2), test_set_resampled="no"))
        metrics.append(dict(technique=label, precision=precision_score(y_te, pred, zero_division=0), recall=recall_score(y_te, pred),
                            f1=f1_score(y_te, pred), balanced_accuracy=balanced_accuracy_score(y_te, pred),
                            roc_auc=roc_auc_score(y_te, prob), pr_auc=average_precision_score(y_te, prob),
                            true_neg=int(tn), false_pos=int(fp), false_neg=int(fn), true_pos=int(tp)))
    m = pd.DataFrame(metrics).round(4)
    note = pd.DataFrame([dict(technique="NOTE", precision=np.nan, recall=np.nan, f1=np.nan, balanced_accuracy=np.nan, roc_auc=np.nan,
                              pr_auc=float(round(y_te.mean(), 4)), true_neg=np.nan, false_pos=np.nan, false_neg=np.nan, true_pos=np.nan)])
    note["technique"] = ("Target: review score <= 2, purchase-time features only, stratified 80/20 split, threshold 0.5. "
                         "PR-AUC of a random model = minority share (shown in pr_auc on this row). Illustrative protocol, not a tuned model.")
    return {"balancing_class_counts": pd.DataFrame(counts), "balancing_metrics": pd.concat([m, note], ignore_index=True)}


# --------------------------------------------------------------------------
def build_analytical(c: dict[str, pd.DataFrame] | None = None, snapshot_date=None) -> dict[str, pd.DataFrame]:
    c = c or load_all_processed()
    sales = build_sales(c)
    return {"sales_orders": sales, "customers": build_customers(c, sales, snapshot_date), "products": build_products(c),
            "sellers": build_sellers(c), "delivery": build_delivery(c, sales)}


def run_imbalance(c: dict[str, pd.DataFrame], tables: dict[str, pd.DataFrame], save: bool = True) -> dict[str, pd.DataFrame]:
    dist = imbalance_distributions(c)
    res = {**imbalance_tables(dist), "concentration": concentration(c), "balancing_guidance": BALANCING_GUIDANCE,
           **evaluate_balancing(tables["sales_orders"], tables["delivery"])}
    if save:
        write_excel(REPORTS_DIR / "imbalance_analysis.xlsx", res, images={"charts": imbalance_chart(dist)})
    return res


def run_feature_engineering(save: bool = True) -> dict:
    c = load_all_processed()
    tables = build_analytical(c)
    if save:
        ANALYTICAL_DIR.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            df.to_csv(ANALYTICAL_DIR / f"{name}.csv", index=False)
        write_excel(REPORTS_DIR / "data_quality_report.xlsx", {"analytical_dictionary": analytical_dictionary(tables)}, mode="a")
    return dict(tables=tables, imbalance=run_imbalance(c, tables, save))


if __name__ == "__main__":
    from .validation import validate_analytical
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60, "display.max_rows", 200)
    out = run_feature_engineering()
    for k, v in out["tables"].items():
        print(f"{k}: {v.shape}")
    im = out["imbalance"]
    print(im["summary"].drop(columns=["interpretation", "ml_recommendation"]).to_string(index=False))
    print(im["concentration"].to_string(index=False))
    print(im["balancing_class_counts"].to_string(index=False))
    print(im["balancing_metrics"].iloc[:-1].to_string(index=False))
    print(validate_analytical().to_string(index=False))
