"""Synthetic repeat purchases for the churn agent.

The real data has almost no returning customers (3.1%), so a churn label is ~99%
positive. This module adds SYNTHETIC follow-up orders for existing customers.

How a synthetic order is made
  * A customer's chance of returning depends on their last order (review score,
    late delivery, delivery time, order value, basket size, payment type, category)
    plus unobserved per-customer noise. The coefficients are DESIGN CHOICES, not
    estimates: a churn model trained on this data learns the pattern written here.
  * A returning customer receives a copy of a real delivered order placed by a
    customer of the same state (real products, sellers, prices, freight, payment
    and review), shifted to a later date.

Rules
  * Real cleaned data is never modified. Synthetic rows are written to their own
    folder and carry is_synthetic = True.
  * Only the churn agent may use them. Revenue, net revenue and demand must use
    real data only, otherwise they would report sales that never happened.

Run:  python -m src.synthetic      (from the olist-analysis/ folder)
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .cleaning import PROCESSED, load_all_processed
from .feature_engineering import build_delivery, build_sales
from .profiling import ROOT

SYNTHETIC_DIR = ROOT.parent / "dataset" / "synthetic dataset"
TABLES = ["orders", "order_items", "payments", "reviews", "customers"]
SEED = 42
TARGET_RETURN_SHARE = 0.30          # share of eligible customers given at least one synthetic order
LAST_PURCHASE_DATE = pd.Timestamp("2018-08-29")
MAX_ROUNDS = 5                      # at most 5 synthetic orders per customer
MIN_GAP_DAYS, MEDIAN_GAP_DAYS, GAP_SIGMA = 7, 55, 0.9
HABIT_CATEGORIES = {"health_beauty", "bed_bath_table", "sports_leisure", "housewares", "pet_shop", "baby",
                    "perfumery", "food", "food_drink", "drinks", "stationery"}

# Effect of the last order on the log-odds of returning (design choices).
COEF = dict(review_score=0.45, late=-0.80, delivery_days=-0.02, log_order_value=0.25, multi_item=0.30,
            credit_card=0.25, habit_category=0.40, already_returned=0.60, noise_sd=0.80)


def _md5(prefix: str, n: int) -> list[str]:
    return [hashlib.md5(f"{prefix}-{i}".encode()).hexdigest() for i in range(n)]


def order_features(c: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-order facts that drive the return probability (real orders)."""
    sales = build_sales(c)
    d = build_delivery(c, sales)
    f = sales[["order_id", "customer_unique_id", "customer_zip_code_prefix", "customer_city", "customer_state",
               "order_purchase_timestamp", "is_delivered", "has_items", "n_items", "order_value", "payment_total",
               "main_payment_type", "review_score"]].merge(
        d[["order_id", "is_late", "delivery_days", "main_category"]], on="order_id", validate="one_to_one")
    return f


def _linear(f: pd.DataFrame) -> np.ndarray:
    """Log-odds contribution of an order's observable facts."""
    return (COEF["review_score"] * (f.review_score.astype(float).fillna(4).values - 4)
            + COEF["late"] * f.is_late.fillna(False).astype(float).values
            + COEF["delivery_days"] * (f.delivery_days.fillna(12).clip(0, 60).values - 12)
            + COEF["log_order_value"] * np.log(f.order_value.fillna(100).clip(lower=5).values / 100)
            + COEF["multi_item"] * (f.n_items.values > 1)
            + COEF["credit_card"] * (f.main_payment_type == "credit_card").values
            + COEF["habit_category"] * f.main_category.isin(HABIT_CATEGORIES).values)


def _draw(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    return rng.random(n), MIN_GAP_DAYS + rng.lognormal(np.log(MEDIAN_GAP_DAYS), GAP_SIGMA, n)


def simulate_returns(feats: pd.DataFrame) -> pd.DataFrame:
    """Decide who returns, when, and which real order each synthetic order copies."""
    pool = feats[feats.is_delivered & feats.has_items & feats.payment_total.notna() & feats.delivery_days.notna()
                 & feats.order_purchase_timestamp.between("2017-01-01", "2018-08-31 23:59:59")].reset_index(drop=True)
    pool_by_state = {s: g.index.values for s, g in pool.groupby("customer_state")}

    # every customer starts from their most recent real order, if that order was a normal delivered one
    last = feats.sort_values("order_purchase_timestamp").drop_duplicates("customer_unique_id", keep="last")
    cur = last[last.order_id.isin(pool.order_id)].reset_index(drop=True)
    n = len(cur)

    rng = np.random.default_rng(SEED)
    noise = rng.normal(0, COEF["noise_sd"], n)
    u1, gap1 = _draw(rng, n)

    # calibrate the intercept so that TARGET_RETURN_SHARE of customers return in round 1 (inside the data window)
    lin1 = _linear(cur) + noise
    in_window = (cur.order_purchase_timestamp + pd.to_timedelta(gap1, unit="D")).values <= np.datetime64(LAST_PURCHASE_DATE)
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        share = ((u1 < 1 / (1 + np.exp(-(mid + lin1)))) & in_window).mean()
        lo, hi = (mid, hi) if share < TARGET_RETURN_SHARE else (lo, mid)
    intercept = (lo + hi) / 2

    out = []
    idx = np.arange(n)                    # position in `cur` of the customers still active
    state = cur.copy()
    for rnd in range(1, MAX_ROUNDS + 1):
        u, gap = (u1, gap1) if rnd == 1 else _draw(rng, len(state))
        logit = intercept + _linear(state) + noise[idx] + (COEF["already_returned"] if rnd > 1 else 0)
        new_ts = state.order_purchase_timestamp.dt.normalize() + pd.to_timedelta(np.floor(gap), unit="D")
        ok = ((u < 1 / (1 + np.exp(-logit))) & (new_ts <= LAST_PURCHASE_DATE)).values
        if not ok.any():
            break
        keep = state[ok]
        donor_pos = np.array([rng.choice(pool_by_state[s]) for s in keep.customer_state])
        donor = pool.iloc[donor_pos].reset_index(drop=True)
        donor_time = donor.order_purchase_timestamp - donor.order_purchase_timestamp.dt.normalize()
        ts = new_ts[ok].reset_index(drop=True) + donor_time
        out.append(pd.DataFrame({
            "customer_unique_id": keep.customer_unique_id.values, "customer_zip_code_prefix": keep.customer_zip_code_prefix.values,
            "customer_city": keep.customer_city.values, "customer_state": keep.customer_state.values,
            "donor_order_id": donor.order_id.values, "new_purchase_timestamp": ts.values, "round": rnd}))
        # the synthetic order becomes the customer's last order for the next round
        state = donor.assign(customer_unique_id=keep.customer_unique_id.values,
                             customer_zip_code_prefix=keep.customer_zip_code_prefix.values,
                             customer_city=keep.customer_city.values, order_purchase_timestamp=ts.values)
        idx = idx[ok]
    plan = pd.concat(out, ignore_index=True)
    plan["order_id"] = _md5("synthetic-order", len(plan))
    plan["customer_id"] = _md5("synthetic-customer", len(plan))
    plan.attrs["intercept"] = intercept
    plan.attrs["eligible_customers"] = n
    return plan


def materialise(plan: pd.DataFrame, c: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Turn the plan into rows with exactly the cleaned-table schemas (+ is_synthetic)."""
    o = c["orders"]
    donor_purchase = plan.donor_order_id.map(o.set_index("order_id").order_purchase_timestamp)
    plan = plan.assign(shift=plan.new_purchase_timestamp.dt.normalize() - donor_purchase.dt.normalize())
    key = plan[["donor_order_id", "order_id", "customer_id", "shift"]].rename(
        columns={"order_id": "new_order_id", "customer_id": "new_customer_id"})

    def copy_rows(table: str, date_cols: list[str]) -> pd.DataFrame:
        df = key.merge(c[table], left_on="donor_order_id", right_on="order_id", how="inner")
        for col in date_cols:
            df[col] = df[col] + df["shift"]
        df["order_id"] = df["new_order_id"]
        return df

    orders = copy_rows("orders", PROCESSED["orders"]["dates"])
    orders["customer_id"] = orders["new_customer_id"]
    items = copy_rows("order_items", ["shipping_limit_date"])
    payments = copy_rows("payments", [])
    r = c["reviews"]
    reviews = key.merge(r[r.is_latest_review_for_order], left_on="donor_order_id", right_on="order_id", how="inner")
    for col in PROCESSED["reviews"]["dates"]:
        reviews[col] = reviews[col] + reviews["shift"]
    reviews["order_id"] = reviews["new_order_id"]
    reviews["review_id"] = _md5("synthetic-review", len(reviews))
    customers = plan[["customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_city", "customer_state"]].copy()

    out = {"orders": orders[list(o.columns) + ["donor_order_id"]], "order_items": items[list(c["order_items"].columns)],
           "payments": payments[list(c["payments"].columns)], "reviews": reviews[list(r.columns)], "customers": customers}
    for df in out.values():
        df.insert(len(df.columns), "is_synthetic", True)
    return {k: v.reset_index(drop=True) for k, v in out.items()}


def load_synthetic() -> dict[str, pd.DataFrame]:
    out = {}
    for name in TABLES:
        meta = PROCESSED[name]
        out[name] = pd.read_csv(SYNTHETIC_DIR / f"synthetic_{name}.csv", parse_dates=meta["dates"],
                                dtype={col: "string" for col in meta["str"] + ["donor_order_id"]})
    return out


def load_combined() -> dict[str, pd.DataFrame]:
    """Real + synthetic tables for the CHURN agent only. Every row carries is_synthetic."""
    c, syn = load_all_processed(), load_synthetic()
    for name in TABLES:
        real = c[name].assign(is_synthetic=False)
        c[name] = pd.concat([real, syn[name].drop(columns="donor_order_id", errors="ignore")], ignore_index=True)
    return c


def validate(c: dict[str, pd.DataFrame], syn: dict[str, pd.DataFrame], plan: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(check, ok, detail=""):
        rows.append(dict(check=check, result="PASS" if bool(ok) else "FAIL", detail=str(detail)))

    o, so = c["orders"], syn["orders"]
    add("synthetic order_ids unique and not colliding with real ones", so.order_id.is_unique and not so.order_id.isin(o.order_id).any(), len(so))
    add("synthetic customer_ids unique and not colliding with real ones",
        syn["customers"].customer_id.is_unique and not syn["customers"].customer_id.isin(c["customers"].customer_id).any())
    add("synthetic review_ids unique and not colliding with real ones",
        syn["reviews"].review_id.is_unique and not syn["reviews"].review_id.isin(c["reviews"].review_id).any())
    add("every synthetic order belongs to an existing real customer", syn["customers"].customer_unique_id.isin(c["customers"].customer_unique_id).all())
    add("orders -> customers 1:1", so.customer_id.isin(syn["customers"].customer_id).all() and len(so) == len(syn["customers"]))
    for t in ("order_items", "payments", "reviews"):
        add(f"{t} -> synthetic orders (no orphans)", syn[t].order_id.isin(so.order_id).all(), len(syn[t]))
    add("every synthetic order has items and a payment", so.order_id.isin(syn["order_items"].order_id).all() and so.order_id.isin(syn["payments"].order_id).all())
    add("items reference real products and sellers", syn["order_items"].product_id.isin(c["products"].product_id).all()
        and syn["order_items"].seller_id.isin(c["sellers"].seller_id).all())
    add("composite keys unique", not syn["order_items"].duplicated(["order_id", "order_item_id"]).any()
        and not syn["payments"].duplicated(["order_id", "payment_sequential"]).any())
    add("purchase dates inside the data window", so.order_purchase_timestamp.max() <= LAST_PURCHASE_DATE + pd.Timedelta(days=1),
        f"{so.order_purchase_timestamp.min().date()} .. {so.order_purchase_timestamp.max().date()}")
    prev = pd.concat([o.merge(c["customers"], on="customer_id")[["customer_unique_id", "order_purchase_timestamp"]],
                      so.merge(syn["customers"], on="customer_id")[["customer_unique_id", "order_purchase_timestamp"]]])
    first_real = o.merge(c["customers"], on="customer_id").groupby("customer_unique_id").order_purchase_timestamp.min()
    syn_ts = so.merge(syn["customers"], on="customer_id")
    add("every synthetic order is later than the customer's first real order",
        (syn_ts.order_purchase_timestamp > syn_ts.customer_unique_id.map(first_real)).all())
    add("all synthetic rows flagged is_synthetic", all(df.is_synthetic.all() for df in syn.values()))
    add("delivery never precedes purchase", not (so.order_delivered_customer_date < so.order_purchase_timestamp).any())
    n_orders = prev.groupby("customer_unique_id").size()
    add("repeat-customer share after augmentation", True, f"{100 * (n_orders > 1).mean():.1f}% (real data alone: "
        f"{100 * (c['customers'].groupby('customer_unique_id').size() > 1).mean():.1f}%)")
    return pd.DataFrame(rows)


def run_synthetic(save: bool = True) -> dict:
    c = load_all_processed()
    plan = simulate_returns(order_features(c))
    syn = materialise(plan, c)
    checks = validate(c, syn, plan)
    if save:
        SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
        for name, df in syn.items():
            df.to_csv(SYNTHETIC_DIR / f"synthetic_{name}.csv", index=False)
    return dict(plan=plan, synthetic=syn, checks=checks)


if __name__ == "__main__":
    pd.set_option("display.width", 220, "display.max_colwidth", 90)
    res = run_synthetic()
    plan = res["plan"]
    print({k: len(v) for k, v in res["synthetic"].items()})
    print("eligible customers", plan.attrs["eligible_customers"], "| customers given synthetic orders", plan.customer_unique_id.nunique(),
          "| intercept", round(plan.attrs["intercept"], 3))
    print("synthetic orders per round", plan["round"].value_counts().sort_index().to_dict())
    print(res["checks"].to_string(index=False))
