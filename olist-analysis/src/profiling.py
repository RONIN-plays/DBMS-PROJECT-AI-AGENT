"""Phase 1 + 2: read-only profiling, data-quality audit and duplicate analysis.

Nothing in this module modifies data. It also hosts the shared project
configuration (paths, schema registry, loaders, text helpers, Excel writer)
that the other modules import.

Run:  python -m src.profiling      (from the olist-analysis/ folder)
Outputs: reports/data_quality_report.xlsx, reports/duplicate_analysis.xlsx
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Shared configuration
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT.parent / "dataset" / "original dataset"   # the 9 Kaggle CSVs, never modified
PROCESSED_DIR = ROOT.parent / "dataset" / "clean dataset"   # cleaned CSVs, inside the original dataset folder
ANALYTICAL_DIR = ROOT / "data" / "analytical"
REPORTS_DIR = ROOT / "reports"

BRAZIL_UF = ["AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG", "PA",
             "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO"]
# Bounding box of Brazil incl. Fernando de Noronha (lng -32.4)
BRAZIL_LAT = (-33.76, 5.28)
BRAZIL_LNG = (-74.00, -32.30)

ORDER_STATUSES = ["created", "approved", "invoiced", "processing", "shipped",
                  "delivered", "unavailable", "canceled"]
PAYMENT_TYPES = ["credit_card", "boleto", "voucher", "debit_card", "not_defined"]

# pk   : primary key (composite where the grain requires it); None = no key
# str  : columns that must stay strings (IDs, postal codes, free text, categories)
# dates: timestamp columns
SCHEMA = {
    "orders": dict(
        file="olist_orders_dataset.csv", pk=["order_id"],
        str=["order_id", "customer_id", "order_status"],
        dates=["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
               "order_delivered_customer_date", "order_estimated_delivery_date"]),
    "order_items": dict(
        file="olist_order_items_dataset.csv", pk=["order_id", "order_item_id"],
        str=["order_id", "product_id", "seller_id"], dates=["shipping_limit_date"]),
    "payments": dict(
        file="olist_order_payments_dataset.csv", pk=["order_id", "payment_sequential"],
        str=["order_id", "payment_type"], dates=[]),
    "reviews": dict(
        file="olist_order_reviews_dataset.csv", pk=["review_id", "order_id"],
        str=["review_id", "order_id", "review_comment_title", "review_comment_message"],
        dates=["review_creation_date", "review_answer_timestamp"]),
    "customers": dict(
        file="olist_customers_dataset.csv", pk=["customer_id"],
        str=["customer_id", "customer_unique_id", "customer_zip_code_prefix",
             "customer_city", "customer_state"], dates=[]),
    "products": dict(
        file="olist_products_dataset.csv", pk=["product_id"],
        str=["product_id", "product_category_name"], dates=[]),
    "sellers": dict(
        file="olist_sellers_dataset.csv", pk=["seller_id"],
        str=["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"], dates=[]),
    "geolocation": dict(
        file="olist_geolocation_dataset.csv", pk=None,
        str=["geolocation_zip_code_prefix", "geolocation_city", "geolocation_state"], dates=[]),
    "category_translation": dict(
        file="product_category_name_translation.csv", pk=["product_category_name"],
        str=["product_category_name", "product_category_name_english"], dates=[]),
}


def load_raw(name: str) -> pd.DataFrame:
    """Load one raw CSV. IDs / postal codes / text are read as strings (keeps leading
    zeros); timestamps are left as strings so that parsing can be audited later."""
    meta = SCHEMA[name]
    return pd.read_csv(RAW_DIR / meta["file"], dtype={c: "string" for c in meta["str"] + meta["dates"]},
                       encoding="utf-8-sig")


def load_all_raw() -> dict[str, pd.DataFrame]:
    return {name: load_raw(name) for name in SCHEMA}


def _normalize_one(v: str) -> str:
    v = unicodedata.normalize("NFKD", v.strip().lower()).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", v).strip()


def normalize_text(s: pd.Series) -> pd.Series:
    """lower-case, strip accents, trim and collapse internal whitespace.
    Works on the distinct values only, so it stays fast on the 1M-row geolocation file."""
    mapping = {v: _normalize_one(v) for v in s.dropna().unique()}
    return s.map(mapping).astype("string")


def write_excel(path: Path, sheets: dict[str, pd.DataFrame], mode: str = "w", images: dict | None = None) -> None:
    """Write several DataFrames to one workbook (frozen header, auto column width).
    `images` maps sheet name -> PNG bytes buffer, embedded on its own sheet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(engine="openpyxl", mode=mode)
    if mode == "a":
        kwargs["if_sheet_exists"] = "replace"
    with pd.ExcelWriter(path, **kwargs) as xw:
        for sheet, df in sheets.items():
            df = df.copy()
            for c in df.columns:  # Excel cannot store tz / pandas NA scalars in object cells
                if df[c].dtype == "string":
                    df[c] = df[c].astype(object).where(df[c].notna(), None)
            df.to_excel(xw, sheet_name=sheet[:31], index=False)
            ws = xw.sheets[sheet[:31]]
            ws.freeze_panes = "A2"
            for i, col in enumerate(df.columns, start=1):
                sample = df[col].head(200).astype(str).str.len().max() if len(df) else 0
                width = min(max(len(str(col)), int(sample or 0)) + 2, 70)
                ws.column_dimensions[ws.cell(1, i).column_letter].width = width
        for sheet, buf in (images or {}).items():
            from openpyxl.drawing.image import Image
            ws = xw.book.create_sheet(sheet[:31])
            buf.seek(0)
            ws.add_image(Image(buf), "A1")


# --------------------------------------------------------------------------
# Phase 1: profiling
# --------------------------------------------------------------------------
def pk_duplicates(df: pd.DataFrame, key: list[str] | None) -> int:
    """Number of rows whose key value already appeared earlier (0 = key is unique)."""
    return int(df.duplicated(subset=key).sum()) if key else 0


def overview(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in tables.items():
        pk = SCHEMA[name]["pk"]
        rows.append(dict(
            dataset=name, file=SCHEMA[name]["file"], rows=len(df), columns=df.shape[1],
            missing_cells=int(df.isna().sum().sum()),
            missing_cells_pct=round(100 * df.isna().sum().sum() / df.size, 3),
            columns_with_missing=int((df.isna().sum() > 0).sum()),
            exact_duplicate_rows=int(df.duplicated().sum()),
            primary_key=" + ".join(pk) if pk else "(none)",
            duplicate_primary_keys=pk_duplicates(df, pk),
            memory_mb=round(df.memory_usage(deep=True).sum() / 1e6, 1)))
    return pd.DataFrame(rows)


def column_profile(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in tables.items():
        for col in df.columns:
            s = df[col]
            is_str = s.dtype == "string"
            parsed_fail = np.nan
            if col in SCHEMA[name]["dates"]:
                parsed = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
                parsed_fail = int((parsed.isna() & s.notna()).sum())
            rows.append(dict(
                dataset=name, column=col, raw_dtype=str(s.dtype),
                intended_type=("datetime" if col in SCHEMA[name]["dates"] else
                               "string" if is_str else str(s.dtype)),
                non_null=int(s.notna().sum()), missing=int(s.isna().sum()),
                missing_pct=round(100 * s.isna().mean(), 3),
                unique=int(s.nunique()), unique_pct=round(100 * s.nunique() / max(len(s), 1), 3),
                is_constant=bool(s.nunique() <= 1),
                leading_trailing_whitespace=int((s != s.str.strip()).sum()) if is_str else np.nan,
                empty_strings=int((s.str.strip() == "").sum()) if is_str else np.nan,
                uppercase_chars=int(s.str.contains(r"[A-Z]", na=False).sum()) if is_str else np.nan,
                non_ascii_chars=int(s.str.contains(r"[^\x00-\x7F]", na=False).sum()) if is_str else np.nan,
                datetime_parse_failures=parsed_fail,
                example=str(s.dropna().iloc[0])[:60] if s.notna().any() else ""))
    return pd.DataFrame(rows)


def numeric_stats(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in tables.items():
        for col in df.select_dtypes("number").columns:
            s = df[col].dropna()
            q = s.quantile([.01, .25, .5, .75, .99])
            rows.append(dict(
                dataset=name, column=col, count=len(s), mean=s.mean(), median=q[.5], std=s.std(),
                min=s.min(), p01=q[.01], p25=q[.25], p75=q[.75], p99=q[.99], max=s.max(),
                skewness=s.skew(), zeros=int((s == 0).sum()), negatives=int((s < 0).sum())))
    return pd.DataFrame(rows).round(4)


def categorical_distributions(tables: dict[str, pd.DataFrame], max_card: int = 80, top: int = 25) -> pd.DataFrame:
    """Value distributions for low-cardinality columns plus top values of city columns."""
    out = []
    for name, df in tables.items():
        for col in df.columns:
            s = df[col]
            is_city = col.endswith("_city")
            if not (s.nunique() <= max_card or is_city) or col in SCHEMA[name]["dates"]:
                continue
            if s.dtype != "string" and s.nunique() > 30:
                continue
            vc = s.value_counts(dropna=False).head(top)
            out.append(pd.DataFrame(dict(
                dataset=name, column=col, value=vc.index.astype(object).map(lambda v: "<missing>" if pd.isna(v) else v),
                count=vc.values, pct=(100 * vc.values / len(s)).round(3),
                note=f"top {top} of {s.nunique()} values" if s.nunique() > top else "")))
    return pd.concat(out, ignore_index=True)


def parse_dates(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return a copy with the schema's date columns parsed (coerce, never drop rows)."""
    d = df.copy()
    for c in SCHEMA[name]["dates"]:
        d[c] = pd.to_datetime(d[c], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    return d


def orders_missing_by_status(orders: pd.DataFrame) -> pd.DataFrame:
    """Missing timestamps per order status - shows that missingness is structural."""
    cols = ["order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date"]
    g = orders.groupby("order_status")[cols].agg(lambda s: int(s.isna().sum()))
    g.columns = ["missing_" + c for c in cols]
    g.insert(0, "orders", orders["order_status"].value_counts())
    return g.sort_values("orders", ascending=False).reset_index()


def audit_issues(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Olist-specific suspicious / inconsistent value checks. Read-only."""
    issues: list[dict] = []

    def add(dataset, column, issue, n, severity, detail=""):
        issues.append(dict(dataset=dataset, column=column, issue=issue, affected_records=int(n),
                           severity=severity, detail=detail))

    # ---- orders
    o = parse_dates(t["orders"], "orders")
    p, a, c, d, e = (o[x] for x in SCHEMA["orders"]["dates"])
    deliv = o.order_status == "delivered"
    add("orders", "order_status", "status values outside the documented set",
        (~o.order_status.isin(ORDER_STATUSES)).sum(), "high")
    add("orders", "order_delivered_customer_date", "status=delivered but delivery date missing", (deliv & d.isna()).sum(), "medium",
        "contradiction between status and timestamp; cannot be imputed reliably")
    add("orders", "order_delivered_carrier_date", "status=delivered but carrier hand-over date missing", (deliv & c.isna()).sum(), "low")
    add("orders", "order_approved_at", "status=delivered but approval timestamp missing", (deliv & a.isna()).sum(), "low")
    add("orders", "order_approved_at", "approval missing (all statuses)", a.isna().sum(), "info",
        "; ".join(f"{k}={v}" for k, v in o[a.isna()].order_status.value_counts().items()))
    add("orders", "order_delivered_customer_date", "non-delivered status but delivery date present", (~deliv & d.notna()).sum(), "low",
        "; ".join(f"{k}={v}" for k, v in o[~deliv & d.notna()].order_status.value_counts().items()))
    add("orders", "order_approved_at", "approved before purchase", (a < p).sum(), "medium")
    add("orders", "order_delivered_carrier_date", "carrier hand-over before approval", (c < a).sum(), "low",
        "plausible operationally (seller ships before payment clears) or timestamp lag")
    add("orders", "order_delivered_carrier_date", "carrier hand-over before purchase", (c < p).sum(), "medium")
    add("orders", "order_delivered_customer_date", "delivered to customer before carrier hand-over", (d < c).sum(), "medium")
    add("orders", "order_delivered_customer_date", "delivered before purchase", (d < p).sum(), "high")
    add("orders", "order_estimated_delivery_date", "estimated delivery before purchase", (e < p).sum(), "high")
    add("orders", "order_purchase_timestamp", "sparse edge months (<100 orders) at start/end of the period",
        (o.groupby(p.dt.to_period("M")).order_id.transform("size") < 100).sum(), "info",
        f"range {p.min()} .. {p.max()}; incomplete months distort trend/forecast analysis")

    # ---- order items
    it = parse_dates(t["order_items"], "order_items")
    add("order_items", "price", "price <= 0", (it.price <= 0).sum(), "high")
    add("order_items", "freight_value", "freight < 0", (it.freight_value < 0).sum(), "high")
    add("order_items", "freight_value", "freight = 0 (free shipping)", (it.freight_value == 0).sum(), "info", "legitimate")
    seq_ok = it.groupby("order_id").order_item_id.agg(lambda s: sorted(s) == list(range(1, len(s) + 1)))
    add("order_items", "order_item_id", "orders whose item sequence is not 1..n", (~seq_ok).sum(), "medium")
    m = it.merge(o[["order_id", "order_purchase_timestamp"]], on="order_id", how="left")
    gap = (m.shipping_limit_date - m.order_purchase_timestamp).dt.days
    add("order_items", "shipping_limit_date", "shipping limit > 60 days after purchase", (gap > 60).sum(), "low",
        f"max gap {gap.max()} days; latest shipping_limit_date {it.shipping_limit_date.max()}")
    add("order_items", "shipping_limit_date", "shipping limit before purchase", (gap < 0).sum(), "low")
    same = it.duplicated(subset=["order_id", "product_id", "seller_id", "price", "freight_value"], keep=False)
    add("order_items", "order_id+product_id", "rows repeating the same product in one order", same.sum(), "info",
        "legitimate: one row per unit purchased (quantity is encoded as repeated rows)")

    # ---- payments
    pay = t["payments"]
    add("payments", "payment_type", "payment_type = not_defined", (pay.payment_type == "not_defined").sum(), "medium",
        f"payment_value sum = {pay.loc[pay.payment_type == 'not_defined', 'payment_value'].sum()}")
    add("payments", "payment_installments", "installments = 0", (pay.payment_installments == 0).sum(), "medium",
        "; ".join(f"{k}={v}" for k, v in pay[pay.payment_installments == 0].payment_type.value_counts().items()))
    add("payments", "payment_value", "payment_value = 0", (pay.payment_value == 0).sum(), "low",
        "; ".join(f"{k}={v}" for k, v in pay[pay.payment_value == 0].payment_type.value_counts().items()))
    add("payments", "payment_value", "payment_value < 0", (pay.payment_value < 0).sum(), "high")
    pseq = pay.groupby("order_id").payment_sequential.agg(lambda s: sorted(s) == list(range(1, len(s) + 1)))
    add("payments", "payment_sequential", "orders whose payment sequence is not 1..n", (~pseq).sum(), "low",
        "sequence gaps - some payment records are absent from the extract")
    item_tot = it.groupby("order_id").agg(items=("price", "sum"), freight=("freight_value", "sum")).sum(axis=1)
    pay_tot = pay.groupby("order_id").payment_value.sum()
    both = pd.concat([item_tot.rename("items"), pay_tot.rename("paid")], axis=1).dropna()
    diff = (both.paid - both["items"]).abs()
    add("payments", "payment_value", "order payment total differs from items+freight by > R$0.05", (diff > 0.05).sum(), "low",
        f"{(diff > 1).sum()} differ by > R$1; max diff R${diff.max():.2f} (instalment interest, vouchers, rounding)")

    # ---- reviews
    r = parse_dates(t["reviews"], "reviews")
    add("reviews", "review_score", "score outside 1..5", (~r.review_score.between(1, 5)).sum(), "high")
    add("reviews", "review_id", "review_id values shared by several rows", r.review_id.duplicated(keep=False).sum(), "info",
        f"{r.review_id.duplicated().sum()} surplus rows; same review attached to several orders")
    add("reviews", "order_id", "orders with more than one review", (r.groupby("order_id").size() > 1).sum(), "info",
        "aggregate to one review per order before joining to orders")
    add("reviews", "review_answer_timestamp", "answer earlier than review creation", (r.review_answer_timestamp < r.review_creation_date).sum(), "medium")
    rm = r.merge(o[["order_id", "order_purchase_timestamp"]], on="order_id", how="left")
    add("reviews", "review_creation_date", "review created before the order was purchased",
        (rm.review_creation_date < rm.order_purchase_timestamp.dt.normalize()).sum(), "medium")

    # ---- customers / sellers
    for name, pre in (("customers", "customer"), ("sellers", "seller")):
        df = t[name]
        add(name, f"{pre}_zip_code_prefix", "postal prefix not exactly 5 digits",
            (~df[f"{pre}_zip_code_prefix"].str.fullmatch(r"\d{5}")).sum(), "high")
        add(name, f"{pre}_state", "state not a valid Brazilian UF", (~df[f"{pre}_state"].isin(BRAZIL_UF)).sum(), "high")
        city = df[f"{pre}_city"]
        bad = city.str.contains(r"[\d@/\\,()]| - ", na=False)
        add(name, f"{pre}_city", "city contains digits, '@', '/', ',' or other non-name characters", bad.sum(), "medium",
            "; ".join(city[bad].unique()[:12]))
        add(name, f"{pre}_city", "city spellings that collapse after text normalisation",
            city.nunique() - normalize_text(city).nunique(), "low")
    cu = t["customers"]
    add("customers", "customer_unique_id", "unique customers appearing with more than one customer_id",
        (cu.groupby("customer_unique_id").size() > 1).sum(), "info", "legitimate: repeat purchasers")

    # ---- products
    pr = t["products"]
    dims = ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
    add("products", "product_category_name", "category missing", pr.product_category_name.isna().sum(), "medium",
        "same rows also lack name length, description length and photo count")
    add("products", "dimensions", "weight/dimensions missing", pr[dims].isna().any(axis=1).sum(), "medium")
    add("products", "product_weight_g", "weight = 0 g (physically impossible)", (pr.product_weight_g == 0).sum(), "medium")
    add("products", "dimensions", "any dimension <= 0", (pr[dims[1:]] <= 0).any(axis=1).sum(), "high")
    add("products", "product_name_lenght / product_description_lenght", "misspelled column names ('lenght')", 2, "low", "rename to *_length")
    known = set(t["category_translation"].product_category_name)
    miss = pr.loc[pr.product_category_name.notna() & ~pr.product_category_name.isin(known), "product_category_name"]
    add("products", "product_category_name", "category has no English translation", len(miss), "medium",
        "; ".join(f"{k}={v}" for k, v in miss.value_counts().items()))
    attr = [c for c in pr.columns if c != "product_id"]
    add("products", "(all attributes)", "products with identical attributes but different product_id",
        pr.dropna(subset=attr).duplicated(subset=attr, keep=False).sum(), "info",
        "cannot be proven to be the same product; retained")

    # ---- geolocation
    g = t["geolocation"]
    outside = ~(g.geolocation_lat.between(*BRAZIL_LAT) & g.geolocation_lng.between(*BRAZIL_LNG))
    add("geolocation", "(all)", "exact duplicate rows", g.duplicated().sum(), "medium")
    add("geolocation", "geolocation_lat/lng", "coordinates outside Brazil's bounding box", outside.sum(), "medium",
        f"lat range {g.geolocation_lat.min():.2f}..{g.geolocation_lat.max():.2f}, lng range {g.geolocation_lng.min():.2f}..{g.geolocation_lng.max():.2f}")
    per_zip = g.groupby("geolocation_zip_code_prefix")
    add("geolocation", "geolocation_zip_code_prefix", "postal prefixes with more than one row", (per_zip.size() > 1).sum(), "info",
        f"{g.geolocation_zip_code_prefix.nunique()} distinct prefixes; median {per_zip.size().median():.0f} rows, max {per_zip.size().max()}")
    add("geolocation", "geolocation_state", "postal prefixes mapped to more than one state", (per_zip.geolocation_state.nunique() > 1).sum(), "medium")
    add("geolocation", "geolocation_city", "postal prefixes with more than one raw city spelling", (per_zip.geolocation_city.nunique() > 1).sum(), "low")
    add("geolocation", "geolocation_city", "city spellings that collapse after text normalisation",
        g.geolocation_city.nunique() - normalize_text(pd.Series(g.geolocation_city.unique())).nunique(), "low",
        f"{g.geolocation_city.nunique()} raw spellings")
    add("geolocation", "geolocation_zip_code_prefix", "postal prefix not exactly 5 digits",
        (~g.geolocation_zip_code_prefix.str.fullmatch(r"\d{5}")).sum(), "high")

    # ---- relationships (headline numbers; full validation lives in src/validation.py)
    add("orders", "order_id", "orders without any item", (~o.order_id.isin(it.order_id)).sum(), "info",
        "; ".join(f"{k}={v}" for k, v in o[~o.order_id.isin(it.order_id)].order_status.value_counts().items()))
    add("orders", "order_id", "orders without any payment record", (~o.order_id.isin(pay.order_id)).sum(), "low")
    add("orders", "order_id", "orders without a review", (~o.order_id.isin(r.order_id)).sum(), "info")
    zips = set(g.geolocation_zip_code_prefix)
    add("customers", "customer_zip_code_prefix", "customers whose postal prefix is absent from geolocation",
        (~cu.customer_zip_code_prefix.isin(zips)).sum(), "low")
    add("sellers", "seller_zip_code_prefix", "sellers whose postal prefix is absent from geolocation",
        (~t["sellers"].seller_zip_code_prefix.isin(zips)).sum(), "low")
    return pd.DataFrame(issues)


def redundancy_checks(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in t.items():
        for col in df.columns:
            if df[col].nunique() <= 1:
                rows.append(dict(dataset=name, column=col, finding="constant column", recommendation="drop"))
        cols = list(df.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                if df[a].dtype == df[b].dtype and df[a].equals(df[b].rename(a)):
                    rows.append(dict(dataset=name, column=f"{a} = {b}", finding="identical columns", recommendation="drop one"))
    o, cu = t["orders"], t["customers"]
    rows += [
        dict(dataset="orders/customers", column="customer_id",
             finding=f"customer_id is 1:1 with order_id ({o.customer_id.nunique()} ids for {len(o)} orders): it is an order-level surrogate, not a person",
             recommendation="keep as join key; use customer_unique_id for customer-level analysis"),
        dict(dataset="customers/sellers", column="*_city, *_state",
             finding="city and state are derivable from the postal prefix via geolocation",
             recommendation="keep (cheap, avoids a join; geolocation does not cover every prefix)"),
        dict(dataset="products", column="product_name_lenght, product_description_lenght, product_photos_qty",
             finding="listing metadata, always missing together with category (same 610 rows)",
             recommendation="keep; low analytical value but harmless"),
        dict(dataset="category_translation", column="product_category_name_english",
             finding="lookup table; fully redundant once merged into products",
             recommendation="merge English name into cleaned products, keep the lookup file"),
        dict(dataset="geolocation", column="(rows)",
             finding=f"{t['geolocation'].duplicated().sum()} exact duplicate rows; many points per postal prefix",
             recommendation="deduplicate and aggregate to one row per prefix"),
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Phase 2: duplicate analysis (still read-only)
# --------------------------------------------------------------------------
def key_checks(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Uniqueness tests for primary, composite and business keys."""
    tests = [
        ("orders", ["order_id"], "primary key", "must be unique"),
        ("orders", ["customer_id"], "alternate key", "1:1 with order_id in Olist"),
        ("order_items", ["order_id", "order_item_id"], "composite primary key", "must be unique"),
        ("order_items", ["order_id"], "foreign key", "repeats legitimately (multi-item orders)"),
        ("order_items", ["order_id", "product_id", "seller_id"], "business key", "repeats legitimately (quantity > 1)"),
        ("payments", ["order_id", "payment_sequential"], "composite primary key", "must be unique"),
        ("payments", ["order_id"], "foreign key", "repeats legitimately (split payments / vouchers)"),
        ("reviews", ["review_id", "order_id"], "composite primary key", "must be unique"),
        ("reviews", ["review_id"], "candidate key", "repeats: one review can cover several orders"),
        ("reviews", ["order_id"], "foreign key", "repeats: an order can be reviewed more than once"),
        ("customers", ["customer_id"], "primary key", "must be unique"),
        ("customers", ["customer_unique_id"], "person identifier", "repeats legitimately (repeat customers)"),
        ("products", ["product_id"], "primary key", "must be unique"),
        ("sellers", ["seller_id"], "primary key", "must be unique"),
        ("category_translation", ["product_category_name"], "primary key", "must be unique"),
        ("category_translation", ["product_category_name_english"], "alternate key", "should be unique"),
        ("geolocation", ["geolocation_zip_code_prefix"], "reference key", "repeats: many sampled points per prefix"),
        ("geolocation", ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"], "business key", "repeated points"),
    ]
    rows = []
    for name, key, kind, rule in tests:
        df = t[name]
        dup_all = df.duplicated(subset=key, keep=False)
        surplus = int(df.duplicated(subset=key).sum())
        must = rule.startswith(("must", "should", "1:1"))
        rows.append(dict(dataset=name, key=" + ".join(key), key_type=kind, expectation=rule,
                         is_unique=surplus == 0, distinct_keys=int(len(df) - surplus),
                         rows_sharing_a_key=int(dup_all.sum()), surplus_rows=surplus,
                         verdict=("OK" if surplus == 0 else "VIOLATION") if must else "legitimate repetition - keep"))
    return pd.DataFrame(rows)


def duplicate_summary(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    decisions = {
        "orders": "no duplicates; nothing to remove",
        "order_items": "no exact duplicates; repeated order_id / product rows are units purchased - keep all",
        "payments": "no exact duplicates; repeated order_id rows are split payments - keep all",
        "reviews": "no exact duplicates; keep all rows, de-duplicate to one review per order only in analytical tables",
        "customers": "no duplicates; repeated customer_unique_id = repeat customers - keep all",
        "products": "no duplicates; nothing to remove",
        "sellers": "no duplicates; nothing to remove",
        "geolocation": "drop exact duplicate rows (carry no information), then aggregate to one row per postal prefix",
        "category_translation": "no duplicates; nothing to remove",
    }
    rows = []
    for name, df in t.items():
        pk = SCHEMA[name]["pk"]
        rows.append(dict(dataset=name, rows=len(df), exact_duplicate_rows=int(df.duplicated().sum()),
                         exact_duplicate_pct=round(100 * df.duplicated().mean(), 3),
                         primary_key=" + ".join(pk) if pk else "(none)",
                         duplicate_primary_keys=pk_duplicates(df, pk), proposed_action=decisions[name]))
    return pd.DataFrame(rows)


def review_duplicates(reviews: pd.DataFrame) -> dict[str, pd.DataFrame]:
    r = reviews
    by_rev = r.groupby("review_id").agg(rows=("order_id", "size"), distinct_orders=("order_id", "nunique"),
                                        distinct_scores=("review_score", "nunique"),
                                        distinct_messages=("review_comment_message", lambda s: s.nunique(dropna=False)))
    by_rev = by_rev[by_rev.rows > 1]
    by_ord = r.groupby("order_id").agg(reviews=("review_id", "nunique"), distinct_scores=("review_score", "nunique"),
                                       min_score=("review_score", "min"), max_score=("review_score", "max"))
    by_ord = by_ord[by_ord.reviews > 1]
    summary = pd.DataFrame([
        dict(metric="review rows", value=len(r)),
        dict(metric="distinct review_id", value=r.review_id.nunique()),
        dict(metric="distinct order_id", value=r.order_id.nunique()),
        dict(metric="review_ids attached to >1 order", value=len(by_rev)),
        dict(metric="...of which every row has identical score", value=int((by_rev.distinct_scores == 1).sum())),
        dict(metric="...of which every row has identical message", value=int((by_rev.distinct_messages == 1).sum())),
        dict(metric="orders with >1 review", value=len(by_ord)),
        dict(metric="...of which the reviews disagree on score", value=int((by_ord.distinct_scores > 1).sum())),
        dict(metric="interpretation", value="content-identical review copied to each order of a multi-order purchase; "
                                            "and re-reviews of one order. Neither is a data-entry duplicate."),
    ])
    return {"reviews_summary": summary.astype({"value": str}),
            "reviews_shared_review_id": by_rev.sort_values("rows", ascending=False).head(500).reset_index(),
            "reviews_multi_per_order": by_ord.sort_values("reviews", ascending=False).head(500).reset_index()}


def geolocation_duplicates(g: pd.DataFrame) -> pd.DataFrame:
    per_zip = g.groupby("geolocation_zip_code_prefix").size()
    dedup = g.drop_duplicates()
    q = per_zip.quantile([.25, .5, .75, .95, .99])
    rows = [
        ("rows", len(g)), ("exact duplicate rows", int(g.duplicated().sum())),
        ("rows after exact de-duplication", len(dedup)),
        ("distinct postal prefixes", g.geolocation_zip_code_prefix.nunique()),
        ("prefixes with a single row", int((per_zip == 1).sum())),
        ("rows per prefix - p25", q[.25]), ("rows per prefix - median", q[.5]), ("rows per prefix - p75", q[.75]),
        ("rows per prefix - p95", q[.95]), ("rows per prefix - p99", q[.99]), ("rows per prefix - max", int(per_zip.max())),
        ("distinct (prefix, lat, lng) points", len(g.drop_duplicates(["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"]))),
        ("rows identical except for city spelling", len(dedup) - len(g.drop_duplicates(
            ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng", "geolocation_state"]))),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def formatting_variants(t: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Categorical values that differ only by case, accents or whitespace."""
    summary, examples = [], []
    targets = [("customers", "customer_city"), ("customers", "customer_state"), ("sellers", "seller_city"),
               ("sellers", "seller_state"), ("geolocation", "geolocation_city"), ("geolocation", "geolocation_state"),
               ("orders", "order_status"), ("payments", "payment_type"), ("products", "product_category_name"),
               ("category_translation", "product_category_name_english")]
    for name, col in targets:
        counts = t[name][col].value_counts()
        raw = pd.Series(counts.index, dtype="string")
        norm = normalize_text(raw)
        summary.append(dict(dataset=name, column=col, raw_distinct=len(raw), normalised_distinct=int(norm.nunique()),
                            redundant_variants=int(len(raw) - norm.nunique()),
                            values_with_whitespace_padding=int((raw != raw.str.strip()).sum()),
                            values_with_accents=int(raw.str.contains(r"[^\x00-\x7F]").sum()),
                            values_with_uppercase=int(raw.str.contains(r"[A-Z]").sum())))
        grp = pd.DataFrame(dict(raw=raw.values, norm=norm.values, rows=counts.values))
        multi = grp[grp.groupby("norm").raw.transform("size") > 1]
        if len(multi):
            ex = multi.groupby("norm").agg(variants=("raw", lambda s: " | ".join(s)), n_variants=("raw", "size"),
                                           rows=("rows", "sum")).sort_values("rows", ascending=False).head(40)
            ex.insert(0, "column", col)
            ex.insert(0, "dataset", name)
            examples.append(ex.reset_index().rename(columns={"norm": "normalised_value"}))
    return {"formatting_variants": pd.DataFrame(summary),
            "variant_examples": pd.concat(examples, ignore_index=True) if examples else pd.DataFrame()}


# --------------------------------------------------------------------------
def run_profiling(tables: dict[str, pd.DataFrame] | None = None, save: bool = True) -> dict[str, pd.DataFrame]:
    """Phase 1 + Phase 2. Returns every report table; optionally writes the workbooks."""
    t = tables or load_all_raw()
    quality = {
        "overview": overview(t),
        "column_profile": column_profile(t),
        "numeric_stats": numeric_stats(t),
        "categorical_distributions": categorical_distributions(t),
        "orders_missing_by_status": orders_missing_by_status(t["orders"]),
        "redundancy": redundancy_checks(t),
        "quality_issues": audit_issues(t),
    }
    dups = {"summary": duplicate_summary(t), "key_checks": key_checks(t),
            **review_duplicates(t["reviews"]),
            "geolocation_duplicates": geolocation_duplicates(t["geolocation"]),
            **formatting_variants(t)}
    if save:
        write_excel(REPORTS_DIR / "data_quality_report.xlsx", quality)
        write_excel(REPORTS_DIR / "duplicate_analysis.xlsx", dups)
    return {**quality, **{f"dup_{k}": v for k, v in dups.items()}}


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 90, "display.max_rows", 200)
    res = run_profiling()
    for key in ["overview", "orders_missing_by_status", "quality_issues", "dup_key_checks", "dup_reviews_summary",
                "dup_geolocation_duplicates", "dup_formatting_variants"]:
        print(f"\n=== {key} ===")
        print(res[key].to_string(index=False))
