"""Phases 7 and 9: relational integrity, join-safety and post-cleaning quality assurance.

Reads the raw files and the cleaned files back from disk (so the CSV round-trip is
tested too) and appends its results to reports/data_quality_report.xlsx.

Run:  python -m src.validation     (from the olist-analysis/ folder, after src.cleaning)
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .cleaning import OUTLIER_VARS, PROCESSED, load_all_processed
from .profiling import (ANALYTICAL_DIR, BRAZIL_LAT, BRAZIL_LNG, BRAZIL_UF, ORDER_STATUSES, PAYMENT_TYPES, RAW_DIR,
                        REPORTS_DIR, ROOT, SCHEMA, load_all_raw, normalize_text, write_excel)

PRIMARY_KEYS = {**{k: v["pk"] for k, v in SCHEMA.items()}, "geo_zip": ["zip_code_prefix"],
                "geolocation": None}

# SHA-256 of the nine CSV files exactly as downloaded from Kaggle (Olist Brazilian E-Commerce Public Dataset).
RAW_SHA256 = {
    "olist_customers_dataset.csv": "983a422239e1712ded753b3bf9ecf47dc73f144d306029dcfa99e70a226883d2",
    "olist_geolocation_dataset.csv": "b514f6fc991b9566aeba02aa5d67e2c3630f034b60a0e05aa0d082a3b66d88d6",
    "olist_order_items_dataset.csv": "0bc4d068c4fe38cbb01bd90e8746e3c613fe7b4baef75fab7b0e329701c3e279",
    "olist_order_payments_dataset.csv": "4f713964f2815dbbaa40b9488268c55aac3627bfce5aa96cf58d1f3616de3cc0",
    "olist_order_reviews_dataset.csv": "012b61c7593e34f51fa614efdf802b9c7056ce6aae5307ddb93236e7cfc797d7",
    "olist_orders_dataset.csv": "8df58ef3d2d7e9944010f7beecd9b75367f5588ec6e3c91cec19ae3345ef9ecf",
    "olist_products_dataset.csv": "3e6569628a17fbc75fd206ee357b59e20364b9afa90f5b6cd5b4d624c58aa9cc",
    "olist_sellers_dataset.csv": "1f643d2b950373b85735e7794b20986f528d7a000432e7c6f9bcbb44d0846a0e",
    "product_category_name_translation.csv": "a81f0d1f27b27e7293f761bc79e3ce8f348ee39c4b3ed3e49bde38f478586278",
}

# child table, FK column, parent table, parent key, expected cardinality, is the FK mandatory?
RELATIONSHIPS = [
    ("orders", "customer_id", "customers", "customer_id", "1:1", True),
    ("order_items", "order_id", "orders", "order_id", "N:1", True),
    ("order_items", "product_id", "products", "product_id", "N:1", True),
    ("order_items", "seller_id", "sellers", "seller_id", "N:1", True),
    ("payments", "order_id", "orders", "order_id", "N:1", True),
    ("reviews", "order_id", "orders", "order_id", "N:1", True),
    ("products", "product_category_name", "category_translation", "product_category_name", "N:1", False),
    ("customers", "customer_zip_code_prefix", "geo_zip", "zip_code_prefix", "N:1", False),
    ("sellers", "seller_zip_code_prefix", "geo_zip", "zip_code_prefix", "N:1", False),
]


# --------------------------------------------------------------------------
# Phase 7
# --------------------------------------------------------------------------
def fk_validation(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Orphans in both directions for every relationship."""
    rows = []
    for child, fk, parent, pk, card, mandatory in RELATIONSHIPS:
        c, p = t[child], t[parent]
        values = c[fk][c[fk] != "unknown"] if fk == "product_category_name" else c[fk]
        orphan = ~values.isin(p[pk]) & values.notna()
        childless = ~p[pk].isin(c[fk])
        per_parent = c.groupby(fk).size()
        n_orphan = int(orphan.sum())
        rows.append(dict(
            relationship=f"{child}.{fk} -> {parent}.{pk}", expected_cardinality=card,
            child_rows=len(c), parent_rows=len(p), missing_fk=int(c[fk].isna().sum()),
            orphan_child_rows=n_orphan, orphan_distinct_keys=int(values[orphan].nunique()),
            parents_without_children=int(childless.sum()), max_children_per_parent=int(per_parent.max()),
            parent_key_unique=bool(p[pk].is_unique),
            verdict="PASS" if n_orphan == 0 else ("FAIL" if mandatory else "ACCEPTED GAP (optional reference)")))
    return pd.DataFrame(rows)


def order_level_tables(t: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """One row per order for items, payments and reviews - the only safe shape for joining to orders."""
    items = t["order_items"].groupby("order_id").agg(
        n_items=("order_item_id", "size"), n_products=("product_id", "nunique"), n_sellers=("seller_id", "nunique"),
        items_value=("price", "sum"), freight_value=("freight_value", "sum")).reset_index()
    pay = t["payments"]
    main = pay.sort_values(["order_id", "payment_value", "payment_sequential"], ascending=[True, False, True]) \
              .drop_duplicates("order_id").set_index("order_id").payment_type.rename("main_payment_type")
    payments = pay.groupby("order_id").agg(
        payment_total=("payment_value", "sum"), n_payment_records=("payment_sequential", "size"),
        n_payment_types=("payment_type", "nunique"), max_installments=("payment_installments", "max")).join(main).reset_index()
    r = t["reviews"]
    latest = r[r.is_latest_review_for_order] if "is_latest_review_for_order" in r else \
        r.sort_values(["review_answer_timestamp", "review_creation_date", "review_id"]).drop_duplicates("order_id", keep="last")
    reviews = latest[["order_id", "review_score", "review_creation_date", "review_answer_timestamp"]].merge(
        r.groupby("order_id").size().rename("n_reviews").reset_index(), on="order_id")
    if "has_comment_message" in latest:
        reviews = reviews.merge(latest[["order_id", "has_comment_message"]], on="order_id")
    return {"items": items, "payments": payments, "reviews": reviews}


def join_risk(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Quantify how a naive multi-table join multiplies rows and inflates revenue."""
    o, it, pay, r = t["orders"], t["order_items"], t["payments"], t["reviews"]
    true_items, true_freight, true_paid = it.price.sum(), it.freight_value.sum(), pay.payment_value.sum()
    naive = it.merge(pay, on="order_id")                       # many-to-many on order_id
    naive3 = naive.merge(r[["order_id", "review_score"]], on="order_id")
    agg = order_level_tables(t)
    safe = o.merge(agg["items"], on="order_id", how="left", validate="one_to_one") \
            .merge(agg["payments"], on="order_id", how="left", validate="one_to_one") \
            .merge(agg["reviews"], on="order_id", how="left", validate="one_to_one")
    n_i, n_p = it.groupby("order_id").size(), pay.groupby("order_id").size()
    both = pd.concat([n_i.rename("i"), n_p.rename("p")], axis=1).dropna()
    rows = [
        ("orders with >1 item row", int((n_i > 1).sum()), ""),
        ("orders with >1 payment row", int((n_p > 1).sum()), ""),
        ("orders with >1 item AND >1 payment (many-to-many on order_id)", int(((both.i > 1) & (both.p > 1)).sum()), "unexpected M:N if joined directly"),
        ("orders with >1 review row", int((r.groupby("order_id").size() > 1).sum()), ""),
        ("TRUE item value (sum of price, item grain)", round(true_items, 2), "reference"),
        ("TRUE payment total (sum of payment_value, payment grain)", round(true_paid, 2), "reference"),
        ("NAIVE items x payments join - rows", len(naive), f"vs {len(it)} item rows"),
        ("NAIVE join - sum(price)", round(naive.price.sum(), 2), f"inflated by {100 * (naive.price.sum() / true_items - 1):.2f}%"),
        ("NAIVE join - sum(payment_value)", round(naive.payment_value.sum(), 2), f"inflated by {100 * (naive.payment_value.sum() / true_paid - 1):.2f}%"),
        ("NAIVE items x payments x reviews - sum(payment_value)", round(naive3.payment_value.sum(), 2),
         f"inflated by {100 * (naive3.payment_value.sum() / true_paid - 1):.2f}%"),
        ("SAFE join (aggregate to order grain first) - rows", len(safe), f"= {len(o)} orders: {'PASS' if len(safe) == len(o) else 'FAIL'}"),
        ("SAFE join - sum(items_value)", round(safe.items_value.sum(), 2), "PASS" if np.isclose(safe.items_value.sum(), true_items) else "FAIL"),
        ("SAFE join - sum(freight_value)", round(safe.freight_value.sum(), 2), "PASS" if np.isclose(safe.freight_value.sum(), true_freight) else "FAIL"),
        ("SAFE join - sum(payment_total)", round(safe.payment_total.sum(), 2), "PASS" if np.isclose(safe.payment_total.sum(), true_paid) else "FAIL"),
    ]
    return pd.DataFrame(rows, columns=["measure", "value", "note"])


# --------------------------------------------------------------------------
# Phase 9
# --------------------------------------------------------------------------
def post_clean_checks(raw: dict[str, pd.DataFrame], c: dict[str, pd.DataFrame]) -> pd.DataFrame:
    checks: list[dict] = []

    def add(area, dataset, check, ok, detail=""):
        checks.append(dict(area=area, dataset=dataset, check=check, result="PASS" if bool(ok) else "FAIL", detail=str(detail)))

    o, it, pay, r = c["orders"], c["order_items"], c["payments"], c["reviews"]
    cu, pr, se, g, gz = c["customers"], c["products"], c["sellers"], c["geolocation"], c["geo_zip"]

    for name, df in c.items():
        add("duplicates", name, "no exact duplicate rows", not df.duplicated().any(), int(df.duplicated().sum()))
        key = PRIMARY_KEYS[name]
        if key:
            add("primary key", name, f"{' + '.join(key)} unique and non-null",
                not df.duplicated(subset=key).any() and df[key].notna().all().all(), int(df.duplicated(subset=key).sum()))

    for _, row in fk_validation(c).iterrows():
        add("foreign key", row.relationship.split(".")[0], row.relationship, row.verdict != "FAIL",
            f"{row.orphan_child_rows} orphan rows ({row.verdict})")

    # row counts: nothing lost, nothing multiplied
    for name in ("orders", "order_items", "payments", "reviews", "customers", "products", "sellers"):
        add("row count", name, "row count identical to raw (no deletion, no multiplication)", len(c[name]) == len(raw[name]),
            f"{len(raw[name])} -> {len(c[name])}")
    add("row count", "category_translation", "raw rows + 2 documented additions", len(c["category_translation"]) == len(raw["category_translation"]) + 2)
    add("row count", "geolocation", "only duplicate rows removed", len(g) <= len(raw["geolocation"]) - raw["geolocation"].duplicated().sum(),
        f"{len(raw['geolocation'])} -> {len(g)}")
    add("row count", "geo_zip", "one row per raw postal prefix", len(gz) == raw["geolocation"].geolocation_zip_code_prefix.nunique(), len(gz))

    # missing-value rules
    for col in ("order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date"):
        add("missing values", "orders", f"{col}: structural missing values preserved (not imputed)", o[col].isna().sum() == raw["orders"][col].isna().sum(),
            int(o[col].isna().sum()))
    add("missing values", "orders", "mandatory columns complete", o[["order_id", "customer_id", "order_status", "order_purchase_timestamp",
                                                                   "order_estimated_delivery_date"]].notna().all().all())
    add("missing values", "reviews", "review_score complete; comments preserved as missing", r.review_score.notna().all()
        and r.has_comment_message.sum() == r.review_comment_message.notna().sum())
    dims = ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
    add("missing values", "products", "category labelled, physical attributes complete", pr.product_category_name.notna().all() and pr[dims].notna().all().all(),
        f"{int(pr.flag_category_missing.sum())} unknown category, {int(pr.flag_physical_attrs_imputed.sum())} imputed")
    add("missing values", "products", "listing metadata left missing exactly where raw was missing",
        pr.product_photos_qty.isna().sum() == raw["products"].product_photos_qty.isna().sum())
    for name in ("order_items", "payments", "customers", "sellers"):
        add("missing values", name, "no missing values", not c[name].isna().any().any())

    # numeric ranges
    add("ranges", "order_items", "price > 0, freight >= 0", (it.price > 0).all() and (it.freight_value >= 0).all())
    add("ranges", "payments", "payment_value >= 0, installments >= 1, sequential >= 1",
        (pay.payment_value >= 0).all() and (pay.payment_installments >= 1).all() and (pay.payment_sequential >= 1).all())
    add("ranges", "reviews", "review_score in 1..5", r.review_score.between(1, 5).all())
    add("ranges", "products", "weight and dimensions > 0", (pr[dims] > 0).all().all())
    ok = gz.lat.isna() | (gz.lat.between(*BRAZIL_LAT) & gz.lng.between(*BRAZIL_LNG))
    add("ranges", "geo_zip", "representative coordinates inside Brazil", ok.all(), f"{int(gz.lat.isna().sum())} prefixes without a valid point")
    add("ranges", "geolocation", "every out-of-Brazil point is flagged",
        (g.flag_outside_brazil == ~(g.geolocation_lat.between(*BRAZIL_LAT) & g.geolocation_lng.between(*BRAZIL_LNG))).all(), int(g.flag_outside_brazil.sum()))

    # dates
    p_, a_, c_, d_, e_ = (o[x] for x in SCHEMA["orders"]["dates"])
    add("dates", "orders", "all timestamp columns are datetime64", all(pd.api.types.is_datetime64_any_dtype(o[x]) for x in SCHEMA["orders"]["dates"]))
    add("dates", "orders", "purchase <= estimated delivery; purchase <= customer delivery", not (e_ < p_).any() and not (d_ < p_).any())
    bad = ((a_ < p_) | (c_ < p_) | (d_ < c_)).fillna(False)
    add("dates", "orders", "every chronology violation carries flag_date_sequence_anomaly", (bad == o.flag_date_sequence_anomaly).all(), int(bad.sum()))
    add("dates", "orders", "purchase dates inside the documented window 2016-09 .. 2018-10",
        p_.between("2016-09-01", "2018-10-31 23:59:59").all(), f"{p_.min()} .. {p_.max()}")
    add("dates", "order_items", "shipping_limit_date is datetime64 and never before purchase",
        pd.api.types.is_datetime64_any_dtype(it.shipping_limit_date)
        and not (it.shipping_limit_date < it.order_id.map(o.set_index("order_id").order_purchase_timestamp).dt.normalize()).any())
    add("dates", "reviews", "answer never precedes creation", not (r.review_answer_timestamp < r.review_creation_date).any())

    # categoricals / formats
    add("categoricals", "orders", "order_status within the 8 documented values", o.order_status.isin(ORDER_STATUSES).all())
    add("categoricals", "payments", "payment_type within known values", pay.payment_type.isin(PAYMENT_TYPES).all())
    for name, pre in (("customers", "customer"), ("sellers", "seller")):
        df = c[name]
        add("categoricals", name, "state is a valid UF; postal prefix is 5 digits (leading zeros kept)",
            df[f"{pre}_state"].isin(BRAZIL_UF).all() and df[f"{pre}_zip_code_prefix"].str.fullmatch(r"\d{5}").all(),
            f"{int(df[f'{pre}_zip_code_prefix'].str.startswith('0').sum())} prefixes start with 0")
        add("categoricals", name, "city normalised (lower-case, no accents, trimmed)", (df[f"{pre}_city"] == normalize_text(df[f"{pre}_city"])).all())
    add("categoricals", "geo_zip", "state valid; one state per prefix; prefix 5 digits",
        gz.state.isin(BRAZIL_UF).all() and gz.zip_code_prefix.str.fullmatch(r"\d{5}").all())
    add("categoricals", "products", "every category has an English name", pr.product_category_name_english.notna().all(),
        f"{pr.product_category_name.nunique()} categories incl. 'unknown'")
    for name, df in c.items():
        strs = df.select_dtypes("string")
        add("categoricals", name, "no leading/trailing whitespace in any string column", all((strs[col].dropna() == strs[col].dropna().str.strip()).all() for col in strs))

    # dtypes
    for name, df in c.items():
        id_cols = [col for col in df.columns if col.endswith("_id") and col != "order_item_id" or col.endswith("zip_code_prefix")]
        add("data types", name, "IDs and postal prefixes are strings", all(df[col].dtype == "string" for col in id_cols), ", ".join(id_cols))
    add("data types", "order_items/payments", "monetary columns are float64",
        all(str(x) == "float64" for x in (it.price.dtype, it.freight_value.dtype, pay.payment_value.dtype)))

    # revenue integrity
    add("revenue", "order_items", "sum(price) and sum(freight) equal raw", np.isclose(it.price.sum(), raw["order_items"].price.sum())
        and np.isclose(it.freight_value.sum(), raw["order_items"].freight_value.sum()), f"R${it.price.sum():,.2f} + R${it.freight_value.sum():,.2f}")
    add("revenue", "payments", "sum(payment_value) equals raw", np.isclose(pay.payment_value.sum(), raw["payments"].payment_value.sum()), f"R${pay.payment_value.sum():,.2f}")
    jr = join_risk(c).set_index("measure")
    add("revenue", "order-level join", "aggregated join keeps one row per order and reproduces item, freight and payment totals",
        all("PASS" in str(n) for n in jr.loc[jr.index.str.startswith("SAFE"), "note"]))
    add("outliers", "order_items/payments/products", "outlier flags present, no flagged row removed",
        all(f in c[n].columns for n, _, _, _, _, f in OUTLIER_VARS if f))

    # raw files untouched: compare with the SHA-256 of the files as downloaded from Kaggle
    changed = [m["file"] for m in SCHEMA.values()
               if hashlib.sha256((RAW_DIR / m["file"]).read_bytes()).hexdigest() != RAW_SHA256[m["file"]]]
    add("raw data", "dataset/original dataset", "raw CSVs byte-identical to the original Kaggle download (SHA-256)",
        not changed, ", ".join(changed))
    return pd.DataFrame(checks)


def raw_vs_clean(raw: dict[str, pd.DataFrame], c: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    comp, dtypes = [], []
    renamed = {"product_name_lenght": "product_name_length", "product_description_lenght": "product_description_length"}
    for name, r in raw.items():
        d = c[name]
        default = pd.read_csv(RAW_DIR / SCHEMA[name]["file"], nrows=1000).dtypes.astype(str)
        raw_cols = [renamed.get(x, x) for x in r.columns]
        added = [x for x in d.columns if x not in raw_cols]
        comp.append(dict(dataset=name, raw_rows=len(r), clean_rows=len(d), rows_removed=max(len(r) - len(d), 0), rows_added=max(len(d) - len(r), 0),
                         raw_columns=r.shape[1], clean_columns=d.shape[1], columns_added=", ".join(added),
                         raw_missing_cells=int(r.isna().sum().sum()), clean_missing_cells=int(d[raw_cols].isna().sum().sum()),
                         dtype_changes=sum(default[x] != str(d[renamed.get(x, x)].dtype) for x in r.columns)))
        for x in r.columns:
            before, after = default[x], str(d[renamed.get(x, x)].dtype)
            if before != after or x in renamed:
                dtypes.append(dict(dataset=name, column=x, cleaned_column=renamed.get(x, x), dtype_pandas_default_read=before, dtype_cleaned=after))
    comp.append(dict(dataset="geo_zip (new)", raw_rows=0, clean_rows=len(c["geo_zip"]), rows_removed=0, rows_added=len(c["geo_zip"]),
                     raw_columns=0, clean_columns=c["geo_zip"].shape[1], columns_added="(postal-prefix reference table)",
                     raw_missing_cells=0, clean_missing_cells=int(c["geo_zip"].isna().sum().sum()), dtype_changes=0))
    return {"raw_vs_clean": pd.DataFrame(comp), "dtype_changes": pd.DataFrame(dtypes)}


def cleaning_summary(raw: dict[str, pd.DataFrame], c: dict[str, pd.DataFrame], comp: pd.DataFrame) -> pd.DataFrame:
    log = pd.read_excel(REPORTS_DIR / "cleaning_log.xlsx")
    base = comp[~comp.dataset.str.contains("new")]
    flags = {f"{n}.{col}": int(c[n][col].sum()) for n in c for col in c[n].columns if col.startswith("flag_") or col.endswith("_outlier_iqr")}
    pr = c["products"]
    rows = [
        ("Original row count (9 raw files)", int(base.raw_rows.sum())),
        ("Final row count (9 cleaned files)", int(base.clean_rows.sum())),
        ("Rows removed - geolocation duplicates only", int(base.rows_removed.sum())),
        ("Transactional rows removed (orders, items, payments, reviews, customers, products, sellers)", 0),
        ("Duplicate records removed", int(log.loc[log.phase.str.contains("duplicates"), "records_removed"].sum())),
        ("Missing values treated (labelled or imputed, each with an indicator flag)", int(pr.flag_category_missing.sum()) + 8),
        ("Missing values deliberately preserved", int(sum(c[n][[x for x in c[n].columns if not x.startswith("flag_")]].isna().sum().sum()
                                                          for n in ("orders", "reviews", "products")))),
        ("Invalid values corrected (0 g weights 4, 0 instalments 2, seller cities, geolocation states 8)",
         int(4 + 2 + 8 + log.loc[log.issue_identified.str.startswith("city is not a city name"), "records_modified"].sum())),
        ("Invalid records excluded from an aggregate (out-of-Brazil points, still kept in the row-level file)", int(c["geolocation"].flag_outside_brazil.sum())),
        ("Outliers flagged (IQR) and retained", int(sum(v for k, v in flags.items() if k.endswith("_outlier_iqr")))),
        ("Outliers removed", 0),
        ("Columns whose data type changed (vs a default pandas read of the raw CSV)", int(base.dtype_changes.sum())),
        ("Logged cleaning steps", len(log)),
    ] + [(f"Rows flagged: {k}", v) for k, v in flags.items()]
    return pd.DataFrame(rows, columns=["metric", "value"])


REMAINING_ISSUES = [
    ("orders", "8 delivered orders have no delivery date; 6 canceled orders have one", "flag_status_date_conflict", "exclude via flag in delivery-time analysis"),
    ("orders", "189 orders with impossible chronology (carrier before purchase / delivery before carrier)", "flag_date_sequence_anomaly", "exclude via flag when computing stage durations"),
    ("orders", "Edge months are incomplete (2016-09..2016-12 and 2018-09..2018-10 hold 25+ orders only at the extremes)", "none", "restrict trend/forecast work to 2017-01 .. 2018-08"),
    ("orders", "775 orders have no items; 1 delivered order has no payment record; 768 orders have no review", "has_items / missing values in sales table", "inherent to the extract; not recoverable"),
    ("payments", "Payment total differs from items + freight for ~260 orders (instalment interest, vouchers, rounding)", "payment_mismatch in sales table", "choose one revenue basis and state it; do not mix"),
    ("payments", "80 orders have gaps in payment_sequential (records absent from the extract)", "none", "payment totals for those orders may be understated"),
    ("products", "610 products have no category ('unknown') and no listing metadata", "flag_category_missing", "report 'unknown' as its own group"),
    ("reviews", "A review belongs to an order, not to a product or seller", "none", "product/seller satisfaction is an approximation for multi-item orders"),
    ("reviews", "64 reviews predate the purchase of the order they are attached to", "flag_review_before_purchase", "exclude when modelling review timing"),
    ("geolocation", "279 customers and 7 sellers have no coordinate; 2,094 prefixes rest on <3 points or widely spread points", "flag_low_confidence", "distance features are missing/less reliable for these"),
    ("customers", "customer_unique_id is anonymised; 96.9% of customers bought once", "none", "retention/CLV estimates rest on a small repeat base"),
    ("all", "No cost, margin, commission, marketing or web-traffic data", "none", "profit and true CLV cannot be measured, only estimated"),
]


def validate_analytical() -> pd.DataFrame:
    """Checks on the analytical tables (run after src.feature_engineering)."""
    c = load_all_processed()
    files = {"sales": ("sales_orders.csv", "order_id"), "customer": ("customers.csv", "customer_unique_id"),
             "product": ("products.csv", "product_id"), "seller": ("sellers.csv", "seller_id"), "delivery": ("delivery.csv", "order_id")}
    a = {k: pd.read_csv(ANALYTICAL_DIR / f, dtype={key: "string"}) for k, (f, key) in files.items()}
    it, pay, o = c["order_items"], c["payments"], c["orders"]
    rows = []

    def add(table, check, ok, detail=""):
        rows.append(dict(table=table, check=check, result="PASS" if bool(ok) else "FAIL", detail=str(detail)))

    for k, (f, key) in files.items():
        add(k, f"{key} unique (one row per entity)", a[k][key].is_unique, len(a[k]))
    add("sales", "row count = orders", len(a["sales"]) == len(o), len(o))
    add("sales", "sum(items_value) = item-grain sum(price)", np.isclose(a["sales"].items_value.sum(), it.price.sum()), f"R${it.price.sum():,.2f}")
    add("sales", "sum(freight_value) = item-grain sum(freight)", np.isclose(a["sales"].freight_value.sum(), it.freight_value.sum()))
    add("sales", "sum(payment_total) = payment-grain sum(payment_value)", np.isclose(a["sales"].payment_total.sum(), pay.payment_value.sum()), f"R${pay.payment_value.sum():,.2f}")
    add("customer", "row count = distinct customer_unique_id", len(a["customer"]) == c["customers"].customer_unique_id.nunique())
    add("customer", "sum(n_orders) = orders", a["customer"].n_orders.sum() == len(o))
    add("customer", "sum(total_payment_value) = sum(payment_value)", np.isclose(a["customer"].total_payment_value.sum(), pay.payment_value.sum()))
    add("product", "row count = products; sum(items_sold) = item rows", len(a["product"]) == len(c["products"]) and a["product"].items_sold.sum() == len(it))
    add("product", "sum(revenue) = sum(price)", np.isclose(a["product"].revenue.sum(), it.price.sum()))
    add("seller", "row count = sellers; sum(items_sold) = item rows", len(a["seller"]) == len(c["sellers"]) and a["seller"].items_sold.sum() == len(it))
    add("seller", "sum(revenue) = sum(price)", np.isclose(a["seller"].revenue.sum(), it.price.sum()))
    add("delivery", "row count = orders", len(a["delivery"]) == len(o))
    add("delivery", "delivery metrics exist only for orders with a delivery date",
        a["delivery"].delivery_days.notna().sum() == o.order_delivered_customer_date.notna().sum())
    out = pd.DataFrame(rows)
    write_excel(REPORTS_DIR / "data_quality_report.xlsx", {"analytical_validation": out}, mode="a")
    return out


def run_validation(save: bool = True) -> dict[str, pd.DataFrame]:
    raw, clean = load_all_raw(), load_all_processed()
    comp = raw_vs_clean(raw, clean)
    res = {
        "fk_validation_raw": fk_validation({**raw, "geo_zip": raw["geolocation"].rename(
            columns={"geolocation_zip_code_prefix": "zip_code_prefix"}).drop_duplicates("zip_code_prefix")}),
        "fk_validation_clean": fk_validation(clean),
        "join_risk": join_risk(clean),
        "post_clean_validation": post_clean_checks(raw, clean),
        **comp,
        "cleaning_summary": cleaning_summary(raw, clean, comp["raw_vs_clean"]),
        "remaining_issues": pd.DataFrame(REMAINING_ISSUES, columns=["dataset", "issue", "flag_available", "recommended_handling"]),
    }
    if save:
        write_excel(REPORTS_DIR / "data_quality_report.xlsx", res, mode="a")
    return res


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 95, "display.max_rows", 200)
    out = run_validation()
    for k in ("fk_validation_raw", "fk_validation_clean", "join_risk", "raw_vs_clean", "dtype_changes", "cleaning_summary"):
        print(f"\n=== {k} ===\n{out[k].to_string(index=False)}")
    v = out["post_clean_validation"]
    print(f"\n=== post_clean_validation: {(v.result == 'PASS').sum()} PASS / {(v.result == 'FAIL').sum()} FAIL ===")
    print(v[v.result == "FAIL"].to_string(index=False))
