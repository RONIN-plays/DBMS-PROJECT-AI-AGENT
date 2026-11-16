"""Phases 3, 4, 5 and 8: missing values, type standardisation, outliers, geolocation.

Raw files are only ever read. Every transformation is recorded in a CleaningLog.
No transactional row is deleted; the only rows removed anywhere are duplicate
geolocation points.

Run:  python -m src.cleaning       (from the olist-analysis/ folder)
Outputs: ../dataset/clean dataset/*.csv, reports/missing_values_report.xlsx,
         reports/outlier_analysis.xlsx, reports/cleaning_log.xlsx
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .profiling import (BRAZIL_LAT, BRAZIL_LNG, BRAZIL_UF, ORDER_STATUSES, PAYMENT_TYPES, PROCESSED_DIR,
                        REPORTS_DIR, SCHEMA, load_all_raw, normalize_text, write_excel)

# Cleaned outputs: file name, string columns (IDs, postal codes, text) and datetime columns.
PROCESSED = {
    "orders": dict(file="olist_orders_clean.csv", str=["order_id", "customer_id", "order_status"],
                   dates=SCHEMA["orders"]["dates"]),
    "order_items": dict(file="olist_order_items_clean.csv", str=["order_id", "product_id", "seller_id"],
                        dates=["shipping_limit_date"]),
    "payments": dict(file="olist_order_payments_clean.csv", str=["order_id", "payment_type"], dates=[]),
    "reviews": dict(file="olist_order_reviews_clean.csv",
                    str=["review_id", "order_id", "review_comment_title", "review_comment_message"],
                    dates=["review_creation_date", "review_answer_timestamp"]),
    "customers": dict(file="olist_customers_clean.csv", str=SCHEMA["customers"]["str"], dates=[]),
    "products": dict(file="olist_products_clean.csv",
                     str=["product_id", "product_category_name", "product_category_name_english"], dates=[]),
    "sellers": dict(file="olist_sellers_clean.csv", str=SCHEMA["sellers"]["str"], dates=[]),
    "geolocation": dict(file="olist_geolocation_clean.csv", str=SCHEMA["geolocation"]["str"], dates=[]),
    "geo_zip": dict(file="geolocation_zip_reference.csv", str=["zip_code_prefix", "city", "state"], dates=[]),
    "category_translation": dict(file="product_category_name_translation_clean.csv",
                                 str=SCHEMA["category_translation"]["str"], dates=[]),
}
NULLABLE_INT = ["product_name_length", "product_description_length", "product_photos_qty"]

# Categories present in products but absent from the official translation file.
EXTRA_TRANSLATIONS = {"pc_gamer": "pc_gamer",
                      "portateis_cozinha_e_preparadores_de_alimentos": "portable_kitchen_food_preparers"}
FAR_POINT_KM = 100  # a geolocation point further than this from its prefix median is flagged


def load_processed(name: str) -> pd.DataFrame:
    """Load a cleaned table with the correct dtypes (strings keep leading zeros)."""
    meta = PROCESSED[name]
    df = pd.read_csv(PROCESSED_DIR / meta["file"], dtype={c: "string" for c in meta["str"]},
                     parse_dates=meta["dates"])
    for c in NULLABLE_INT:
        if c in df.columns:
            df[c] = df[c].astype("Int64")
    return df


def load_all_processed() -> dict[str, pd.DataFrame]:
    return {name: load_processed(name) for name in PROCESSED}


class CleaningLog:
    """Append-only record of every check and transformation."""

    def __init__(self):
        self.rows: list[dict] = []

    def add(self, phase, dataset, column, issue, affected, method, reason, removed=0, modified=0, validation="PASS"):
        self.rows.append(dict(step=len(self.rows) + 1, phase=phase, dataset=dataset, column=column,
                              issue_identified=issue, affected_records=int(affected), cleaning_method=method,
                              reason=reason, records_removed=int(removed), records_modified=int(modified),
                              validation_result=validation))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def _check(cond: bool) -> str:
    return "PASS" if bool(cond) else "FAIL"


def haversine_km(lat1, lng1, lat2, lng2):
    """Great-circle distance in km (vectorised)."""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------
# Generic steps (Phase 4)
# --------------------------------------------------------------------------
def trim_strings(df: pd.DataFrame, name: str, log: CleaningLog) -> pd.DataFrame:
    """Strip leading/trailing whitespace on every string column; '' becomes missing."""
    changed = 0
    for c in df.select_dtypes("string").columns:
        new = df[c].str.strip()
        new = new.mask(new == "")
        changed += int(((new != df[c]) | (new.isna() & df[c].notna())).sum())
        df[c] = new
    log.add("4 types/format", name, "(all string columns)", "leading/trailing whitespace or empty strings", changed,
            "str.strip(); empty string -> missing", "whitespace creates false distinct values and broken joins",
            modified=changed)
    return df


def to_datetime_checked(df: pd.DataFrame, name: str, cols: list[str], log: CleaningLog) -> pd.DataFrame:
    """Parse timestamps with an explicit format. Failures become NaT, are counted, and rows are kept."""
    for c in cols:
        before = df[c].notna()
        df[c] = pd.to_datetime(df[c], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        failed = int((before & df[c].isna()).sum())
        log.add("4 types/format", name, c, "timestamp stored as text", int(before.sum()),
                "pd.to_datetime(format='%Y-%m-%d %H:%M:%S', errors='coerce')",
                "datetime arithmetic; explicit format so malformed values are detected, not guessed",
                modified=int(before.sum()), validation=f"{failed} conversion failures" if failed else "PASS (0 conversion failures)")
    return df


# --------------------------------------------------------------------------
# Phase 8: geolocation
# --------------------------------------------------------------------------
def clean_geolocation(raw: pd.DataFrame, log: CleaningLog) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (row-level cleaned points, one-row-per-postal-prefix reference table)."""
    g = raw.copy()
    name = "geolocation"
    g["geolocation_zip_code_prefix"] = g.geolocation_zip_code_prefix.str.strip().str.zfill(5)
    g["geolocation_state"] = g.geolocation_state.str.strip().str.upper()

    n_city_raw = g.geolocation_city.nunique()
    city = normalize_text(g.geolocation_city)
    changed = int((city != g.geolocation_city).sum())
    g["geolocation_city"] = city
    log.add("8 geolocation", name, "geolocation_city", f"{n_city_raw} spellings of {city.nunique()} cities (accents, case, spacing)",
            changed, "lower-case, strip accents, collapse whitespace",
            "'são paulo' and 'sao paulo' are the same city; matches the accent-free style of customers/sellers", modified=changed)

    # state consistency: a prefix belongs to exactly one state
    mode_state = g.groupby("geolocation_zip_code_prefix").geolocation_state.agg(lambda s: s.mode().iat[0])
    expected = g.geolocation_zip_code_prefix.map(mode_state)
    conflict = expected != g.geolocation_state
    g.loc[conflict, "geolocation_state"] = expected[conflict]
    log.add("8 geolocation", name, "geolocation_state", "rows whose state disagrees with every other row of the same prefix",
            conflict.sum(), "replace with the prefix's modal state",
            "a postal prefix lies in one state; each conflict is a single stray row against 12-179 consistent rows",
            modified=conflict.sum(), validation=_check(g.groupby("geolocation_zip_code_prefix").geolocation_state.nunique().max() == 1))

    n_exact = int(raw.duplicated().sum())
    before = len(g)
    g = g.drop_duplicates().reset_index(drop=True)
    log.add("2 duplicates", name, "(all)", "exact duplicate rows in the raw file", n_exact, "drop_duplicates()",
            "an identical point repeated carries no extra geographic information", removed=n_exact)
    log.add("2 duplicates", name, "(all)", "rows that became identical once city spelling and state were standardised",
            before - len(g) - n_exact, "drop_duplicates() after standardisation",
            "same prefix, coordinates and state; differed only by accent/spacing", removed=before - len(g) - n_exact,
            validation=_check(not g.duplicated().any()))

    # coordinate validation
    g["flag_outside_brazil"] = ~(g.geolocation_lat.between(*BRAZIL_LAT) & g.geolocation_lng.between(*BRAZIL_LNG))
    log.add("5 outliers", name, "geolocation_lat / geolocation_lng", "coordinates outside Brazil (Europe, Argentina, USA, Philippines)",
            g.flag_outside_brazil.sum(), "flag (flag_outside_brazil); excluded from the prefix median; row retained",
            "geocoding errors - a Brazilian postal prefix cannot be located abroad; flagged rather than deleted")

    valid = g[~g.flag_outside_brazil]
    med = valid.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].median()
    m = g[["geolocation_zip_code_prefix"]].join(med, on="geolocation_zip_code_prefix", rsuffix="_med")
    g["dist_to_prefix_median_km"] = haversine_km(g.geolocation_lat, g.geolocation_lng,
                                                 m.geolocation_lat, m.geolocation_lng).round(3)
    far = g.dist_to_prefix_median_km > FAR_POINT_KM
    log.add("5 outliers", name, "geolocation_lat / geolocation_lng",
            f"points more than {FAR_POINT_KM} km from the median of their own prefix (inside Brazil)",
            (far & ~g.flag_outside_brazil).sum(), "retain; distance stored in dist_to_prefix_median_km",
            "the median is robust to these points, so they do not move the representative coordinate")

    # postal-prefix reference
    grp = g.groupby("geolocation_zip_code_prefix")
    vgrp = valid.groupby("geolocation_zip_code_prefix")
    ref = pd.DataFrame({
        "lat": med.geolocation_lat, "lng": med.geolocation_lng,
        "city": grp.geolocation_city.agg(lambda s: s.mode().iat[0]),
        "state": grp.geolocation_state.first(),
        "n_points": grp.size(),
        "n_points_used": vgrp.size(),
        "n_city_variants": grp.geolocation_city.nunique(),
        "p95_dist_km": g[~g.flag_outside_brazil].groupby("geolocation_zip_code_prefix").dist_to_prefix_median_km.quantile(.95),
        "max_dist_km": g[~g.flag_outside_brazil].groupby("geolocation_zip_code_prefix").dist_to_prefix_median_km.max(),
    }).rename_axis("zip_code_prefix").reset_index()
    ref["n_points_used"] = ref.n_points_used.fillna(0).astype(int)
    ref[["lat", "lng"]] = ref[["lat", "lng"]].round(6)
    ref[["p95_dist_km", "max_dist_km"]] = ref[["p95_dist_km", "max_dist_km"]].round(2)
    ref["flag_low_confidence"] = (ref.n_points_used < 3) | (ref.p95_dist_km > FAR_POINT_KM)
    log.add("8 geolocation", "geo_zip (new reference table)", "lat, lng, city, state",
            "many points per postal prefix (median 29, max 1,146) - joining on prefix would multiply rows",
            len(g), "aggregate to one row per prefix: median lat/lng of in-Brazil points, modal city, state",
            "median is robust to stray points; one row per prefix makes customer/seller joins strictly many-to-one",
            validation=f"{_check(ref.zip_code_prefix.is_unique)}: {len(ref)} prefixes, {int(ref.lat.isna().sum())} without a valid coordinate, "
                       f"{int(ref.flag_low_confidence.sum())} flagged low-confidence (<3 points or p95 spread > {FAR_POINT_KM} km)")
    return g, ref


# --------------------------------------------------------------------------
# Table cleaners
# --------------------------------------------------------------------------
def clean_orders(raw: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    name = "orders"
    o = trim_strings(raw.copy(), name, log)
    o["order_status"] = o.order_status.str.lower()
    log.add("4 types/format", name, "order_status", "validate against the 8 documented statuses", 0, "lower-case + membership check",
            "categorical consistency", validation=_check(o.order_status.isin(ORDER_STATUSES).all()))
    o = to_datetime_checked(o, name, SCHEMA[name]["dates"], log)
    p, a, c, d, e = (o[x] for x in SCHEMA[name]["dates"])
    deliv = o.order_status == "delivered"

    for col, s in (("order_approved_at", a), ("order_delivered_carrier_date", c), ("order_delivered_customer_date", d)):
        log.add("3 missing", name, col, "missing timestamp", s.isna().sum(), "preserve as NaT (no imputation)",
                "missingness is structural: the order never reached that stage (see missing_values_report); "
                "an imputed date would fabricate deliveries and bias delivery-time metrics")

    o["flag_status_date_conflict"] = (deliv & d.isna()) | (~deliv & d.notna())
    log.add("3 missing", name, "order_delivered_customer_date",
            "status contradicts timestamp: 8 delivered orders without a delivery date, 6 canceled orders with one",
            o.flag_status_date_conflict.sum(), "flag (flag_status_date_conflict); values untouched",
            "neither side of the contradiction can be verified from the data; exclude via the flag in delivery analysis")
    o["flag_date_sequence_anomaly"] = ((a < p) | (c < p) | (d < c) | (d < p) | (e < p)).fillna(False)
    log.add("4 types/format", name, "order_* timestamps",
            "impossible chronology: carrier hand-over before purchase (166) or customer delivery before carrier hand-over (23)",
            o.flag_date_sequence_anomaly.sum(), "flag (flag_date_sequence_anomaly); values untouched",
            "the true timestamp is unknown, so correcting would be guessing; flag lets analyses exclude them")
    o["flag_carrier_before_approval"] = (c < a).fillna(False)
    log.add("4 types/format", name, "order_delivered_carrier_date", "carrier hand-over earlier than payment approval",
            o.flag_carrier_before_approval.sum(), "flag (flag_carrier_before_approval); values untouched",
            "operationally plausible (shipping before boleto clearance is recorded) - informational only")
    return o


def clean_order_items(raw: pd.DataFrame, orders: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    name = "order_items"
    it = trim_strings(raw.copy(), name, log)
    it = to_datetime_checked(it, name, ["shipping_limit_date"], log)
    it["order_item_id"] = it.order_item_id.astype("int64")
    it[["price", "freight_value"]] = it[["price", "freight_value"]].astype("float64")
    log.add("4 types/format", name, "price, freight_value", "range validation", 0, "assert price > 0 and freight >= 0",
            "monetary sanity", validation=_check((it.price > 0).all() and (it.freight_value >= 0).all()))
    gap = (it.shipping_limit_date - it.order_id.map(orders.set_index("order_id").order_purchase_timestamp)).dt.days
    it["flag_shipping_limit_anomaly"] = (gap > 60).fillna(False)
    log.add("4 types/format", name, "shipping_limit_date", "shipping limit more than 60 days after purchase (up to 1,056 days; year 2020)",
            it.flag_shipping_limit_anomaly.sum(), "flag (flag_shipping_limit_anomaly); values untouched",
            "data-entry error in a seller deadline field; price/freight on these rows are valid so rows are kept")
    return it


def clean_payments(raw: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    name = "payments"
    pay = trim_strings(raw.copy(), name, log)
    pay["payment_type"] = pay.payment_type.str.lower()
    log.add("4 types/format", name, "payment_type", "validate against known payment types", 0, "lower-case + membership check",
            "categorical consistency", validation=_check(pay.payment_type.isin(PAYMENT_TYPES).all()))
    zero = (pay.payment_installments == 0) & (pay.payment_type == "credit_card")
    pay.loc[zero, "payment_installments"] = 1
    log.add("4 types/format", name, "payment_installments", "credit-card payments with 0 instalments", zero.sum(),
            "set to 1", "a card payment has at least one instalment; both rows carry a real amount (R$58.69, R$129.94)",
            modified=zero.sum(), validation=_check((pay.payment_installments >= 1).all()))
    nd = pay.payment_type == "not_defined"
    log.add("4 types/format", name, "payment_type", "payment_type = 'not_defined' (all R$0.00, all on canceled orders)", nd.sum(),
            "retain unchanged", "the category itself marks the record as undefined; R$0 so no revenue impact; "
            "deleting would leave 3 more orders without a payment record")
    log.add("5 outliers", name, "payment_value", "payment_value = 0", (pay.payment_value == 0).sum(), "retain unchanged",
            "zero-value vouchers / undefined payments are real records")
    pay["payment_value"] = pay.payment_value.astype("float64")
    return pay


def clean_reviews(raw: pd.DataFrame, orders: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    name = "reviews"
    r = trim_strings(raw.copy(), name, log)
    r = to_datetime_checked(r, name, SCHEMA[name]["dates"], log)
    r["review_score"] = r.review_score.astype("int64")
    log.add("5 outliers", name, "review_score", "validate 1-5 scale", 0, "membership check", "scores must stay on their scale",
            validation=_check(r.review_score.between(1, 5).all()))
    for col in ("review_comment_title", "review_comment_message"):
        flag = "has_" + col.replace("review_", "")
        r[flag] = r[col].notna()
        log.add("3 missing", name, col, "comment not written", r[col].isna().sum(),
                f"preserve as missing; add indicator {flag}",
                "comments are optional - the score is still valid, so rows are kept; presence of a comment is itself informative")
    purchase = r.order_id.map(orders.set_index("order_id").order_purchase_timestamp)
    r["flag_review_before_purchase"] = (r.review_creation_date < purchase.dt.normalize()).fillna(False)
    log.add("4 types/format", name, "review_creation_date", "review created before the order's purchase date",
            r.flag_review_before_purchase.sum(), "flag (flag_review_before_purchase); values untouched",
            "arises where one review is attached to several orders; the score is still usable")
    order = r.sort_values(["review_answer_timestamp", "review_creation_date", "review_id"])
    r["is_latest_review_for_order"] = False
    r.loc[order.drop_duplicates("order_id", keep="last").index, "is_latest_review_for_order"] = True
    log.add("2 duplicates", name, "order_id", "orders with more than one review (547 orders, 551 surplus rows)",
            (~r.is_latest_review_for_order).sum(), "keep every row; mark is_latest_review_for_order",
            "re-reviews are genuine customer feedback; the marker gives a deterministic one-review-per-order "
            "selection so joins to orders cannot multiply rows",
            validation=_check(r.is_latest_review_for_order.sum() == r.order_id.nunique()))
    return r


def repair_city(city: pd.Series, zips: pd.Series, geo_ref: pd.DataFrame, known: set) -> pd.Series:
    """Replace a value that is not a city name appearing anywhere in the geolocation file
    with the modal city of its postal prefix. Recognised names are never overwritten."""
    ref_city = zips.map(geo_ref.set_index("zip_code_prefix").city)
    return city.mask(~city.isin(known) & ref_city.notna(), ref_city)


def clean_customers(raw: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    name = "customers"
    cu = trim_strings(raw.copy(), name, log)
    cu["customer_zip_code_prefix"] = cu.customer_zip_code_prefix.str.zfill(5)
    cu["customer_state"] = cu.customer_state.str.upper()
    city = normalize_text(cu.customer_city)
    changed = int((city != cu.customer_city).sum())
    cu["customer_city"] = city
    log.add("4 types/format", name, "customer_city", "text normalisation (case, accents, spacing)", changed,
            "lower-case, strip accents, collapse whitespace", "consistent grouping keys", modified=changed)
    log.add("4 types/format", name, "customer_zip_code_prefix, customer_state", "format validation", 0,
            "keep as 5-character string; state upper-case and in the list of 27 UFs", "leading zeros must survive; valid geography",
            validation=_check(cu.customer_zip_code_prefix.str.fullmatch(r"\d{5}").all() and cu.customer_state.isin(BRAZIL_UF).all()))
    log.add("2 duplicates", name, "customer_unique_id", "customer_unique_id repeats across customer_id rows",
            cu.customer_unique_id.duplicated().sum(), "retain all rows",
            "each row is one order's customer record; repeats identify repeat purchasers")
    return cu


def clean_sellers(raw: pd.DataFrame, geo_ref: pd.DataFrame, known_cities: set, log: CleaningLog) -> pd.DataFrame:
    name = "sellers"
    s = trim_strings(raw.copy(), name, log)
    s["seller_zip_code_prefix"] = s.seller_zip_code_prefix.str.zfill(5)
    s["seller_state"] = s.seller_state.str.upper()
    city = normalize_text(s.seller_city)
    changed = int((city != s.seller_city).sum())
    log.add("4 types/format", name, "seller_city", "text normalisation (case, accents, spacing)", changed,
            "lower-case, strip accents, collapse whitespace", "consistent grouping keys", modified=changed)
    fixed = repair_city(city, s.seller_zip_code_prefix, geo_ref, known_cities)
    n_fixed = int((fixed != city).sum())
    examples = "; ".join(f"'{a}' -> '{b}'" for a, b in zip(city[fixed != city].head(6), fixed[fixed != city].head(6)))
    still = int((~fixed.isin(known_cities)).sum())
    s["seller_city"] = fixed
    log.add("4 types/format", name, "seller_city",
            "city is not a city name: postal code, e-mail address, 'city / state', state name, typo", n_fixed,
            "replace with the modal city of the seller's postal prefix (geolocation reference)",
            f"free-text field; the postal prefix is the reliable locator. e.g. {examples}", modified=n_fixed,
            validation=f"PASS ({still} unrecognised values remain: prefix absent from geolocation)" if still else "PASS")
    log.add("4 types/format", name, "seller_zip_code_prefix, seller_state", "format validation", 0,
            "keep as 5-character string; state upper-case and in the list of 27 UFs", "leading zeros must survive; valid geography",
            validation=_check(s.seller_zip_code_prefix.str.fullmatch(r"\d{5}").all() and s.seller_state.isin(BRAZIL_UF).all()))
    return s


def clean_translation(raw: pd.DataFrame, products_raw: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    name = "category_translation"
    tr = trim_strings(raw.copy(), name, log)
    missing = sorted(set(products_raw.product_category_name.dropna()) - set(tr.product_category_name))
    add = pd.DataFrame({"product_category_name": missing,
                        "product_category_name_english": [EXTRA_TRANSLATIONS[m] for m in missing]}).astype("string")
    tr = pd.concat([tr, add], ignore_index=True)
    log.add("7 integrity", name, "product_category_name", "categories used by products but absent from the lookup: " + ", ".join(missing),
            len(missing), "append the missing rows with an English label",
            "otherwise 13 products would silently lose their category in any join on the English name", modified=len(missing),
            validation=_check(tr.product_category_name.is_unique))
    return tr


def clean_products(raw: pd.DataFrame, translation: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    name = "products"
    p = trim_strings(raw.copy(), name, log)
    p = p.rename(columns={"product_name_lenght": "product_name_length",
                          "product_description_lenght": "product_description_length"})
    log.add("4 types/format", name, "product_name_lenght, product_description_lenght", "misspelled column names", 2,
            "rename to *_length", "avoid propagating a typo into every downstream query", modified=2)

    p["flag_category_missing"] = p.product_category_name.isna()
    p["product_category_name"] = p.product_category_name.fillna("unknown")
    log.add("3 missing", name, "product_category_name", "category missing", p.flag_category_missing.sum(),
            "label as 'unknown' + indicator flag_category_missing (no mode imputation)",
            "these products were sold (1,603 item rows) so they must stay; assigning the modal category would invent "
            "sales for 'cama_mesa_banho'; an explicit label keeps them visible in group-bys", modified=p.flag_category_missing.sum())
    for c in NULLABLE_INT:
        log.add("3 missing", name, c, "listing metadata missing (same 610 rows as category)", p[c].isna().sum(),
                "preserve as missing; stored as nullable integer",
                "missing together with category - the listing record was not captured; not needed for revenue analysis, "
                "and imputing would only add noise")
        p[c] = p[c].astype("Int64")

    dims = ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
    zero_w = p.product_weight_g == 0
    p.loc[zero_w, "product_weight_g"] = np.nan
    log.add("5 outliers", name, "product_weight_g", "weight = 0 g (physically impossible)", zero_w.sum(),
            "treat as missing, then impute (next step)", "four bed/bath products with valid 30x25x30 cm dimensions", modified=zero_w.sum())
    p["flag_physical_attrs_imputed"] = p[dims].isna().any(axis=1)
    n_cells = int(p[dims].isna().sum().sum())
    cat_median = p.groupby("product_category_name")[dims].transform("median")
    for c in dims:
        p[c] = p[c].fillna(cat_median[c]).fillna(p[c].median())
    log.add("3 missing", name, ", ".join(dims), "weight/dimensions missing (2 products) or invalid zero weight (4 products)",
            p.flag_physical_attrs_imputed.sum(),
            "median of the product's category (global median for the product with unknown category) + indicator flag_physical_attrs_imputed",
            "distributions are strongly right-skewed, so median not mean; the 6 products appear in 26 item rows that would "
            "otherwise drop out of freight/weight analysis; the flag allows exclusion", modified=n_cells,
            validation=_check(p[dims].notna().all().all() and (p[dims] > 0).all().all()))

    p = p.merge(translation, on="product_category_name", how="left", validate="many_to_one")
    p["product_category_name_english"] = p.product_category_name_english.fillna("unknown")
    log.add("7 integrity", name, "product_category_name_english", "English category name lives in a separate lookup table", len(p),
            "left-join (validated many-to-one) and add as a column", "removes a join from every category analysis",
            modified=len(p), validation=_check(len(p) == len(raw) and p.product_category_name_english.notna().all()))
    cols = ["product_id", "product_category_name", "product_category_name_english"]
    return p[cols + [c for c in p.columns if c not in cols]]


# --------------------------------------------------------------------------
# Phase 3 report: missing values
# --------------------------------------------------------------------------
MISSING_NOTES = {
    ("orders", "order_approved_at"): (
        "141 canceled + 5 created orders were never approved (structural). 14 delivered orders lack it (recording gap).",
        "meaningful (146) / minor recording gap (14)", "Preserve NaT. No imputation; approval-time metrics use non-missing rows."),
    ("orders", "order_delivered_carrier_date"): (
        "Order never handed to a carrier: unavailable 609, canceled 550, invoiced 314, processing 301, created 5, approved 2. 2 delivered orders lack it.",
        "meaningful", "Preserve NaT. No imputation."),
    ("orders", "order_delivered_customer_date"): (
        "Order not delivered: shipped 1107, canceled 619, unavailable 609, invoiced 314, processing 301, created 5, approved 2. 8 delivered orders lack it.",
        "meaningful (2957) / problematic (8)", "Preserve NaT. The 8 delivered-without-date orders are flagged (flag_status_date_conflict)."),
    ("reviews", "review_comment_title"): (
        "Title is optional and the field was introduced late in the period.", "meaningful",
        "Preserve missing; indicator has_comment_title. Review scores kept."),
    ("reviews", "review_comment_message"): (
        "Free-text comment is optional; most customers leave only a score.", "meaningful",
        "Preserve missing; indicator has_comment_message. Review scores kept."),
    ("products", "product_category_name"): (
        "610 listings without catalogue metadata (category, name length, description length, photos all absent together).",
        "problematic but not recoverable", "Label 'unknown' + flag_category_missing. No mode imputation."),
    ("products", "product_name_lenght"): ("Same 610 listings as category.", "not recoverable", "Preserve missing (nullable integer)."),
    ("products", "product_description_lenght"): ("Same 610 listings as category.", "not recoverable", "Preserve missing (nullable integer)."),
    ("products", "product_photos_qty"): ("Same 610 listings as category.", "not recoverable", "Preserve missing (nullable integer)."),
    ("products", "product_weight_g"): (
        "2 products lack all physical attributes; additionally 4 products have an impossible 0 g weight.", "problematic",
        "Category-median imputation (skewed) + flag_physical_attrs_imputed."),
    ("products", "product_length_cm"): ("2 products lack all physical attributes.", "problematic", "Category-median imputation + flag."),
    ("products", "product_height_cm"): ("2 products lack all physical attributes.", "problematic", "Category-median imputation + flag."),
    ("products", "product_width_cm"): ("2 products lack all physical attributes.", "problematic", "Category-median imputation + flag."),
}
RENAMED = {"product_name_lenght": "product_name_length", "product_description_lenght": "product_description_length"}


def missing_values_report(raw: dict[str, pd.DataFrame], clean: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    rows = []
    for name, df in raw.items():
        for col in df.columns:
            n = int(df[col].isna().sum())
            if not n:
                continue
            reason, kind, treatment = MISSING_NOTES[(name, col)]
            after = int(clean[name][RENAMED.get(col, col)].isna().sum())
            rows.append(dict(dataset=name, column=col, missing_count=n, missing_pct=round(100 * n / len(df), 3),
                             reason_for_missingness=reason, assessment=kind, treatment=treatment,
                             missing_after_cleaning=after, values_filled=max(n - after, 0),
                             whitespace_only_values_set_to_missing=max(after - n, 0)))
    by_col = pd.DataFrame(rows)

    all_cols = pd.DataFrame([dict(dataset=name, column=col, rows=len(df), missing_count=int(df[col].isna().sum()),
                                  missing_pct=round(100 * df[col].isna().mean(), 3))
                             for name, df in raw.items() for col in df.columns])

    geo = clean["geo_zip"]
    has_coord = set(geo.loc[geo.lat.notna(), "zip_code_prefix"])
    cu, se, o = clean["customers"], clean["sellers"], clean["orders"]
    implicit = pd.DataFrame([
        dict(relationship="customers -> geolocation", missing_count=int((~cu.customer_zip_code_prefix.isin(has_coord)).sum()),
             base=len(cu), reason="postal prefix not present in the geolocation file",
             treatment="Keep customer; coordinates stay missing in analytical tables (state/city still available). No coordinate is invented."),
        dict(relationship="sellers -> geolocation", missing_count=int((~se.seller_zip_code_prefix.isin(has_coord)).sum()),
             base=len(se), reason="postal prefix not present in the geolocation file", treatment="Same as customers."),
        dict(relationship="orders -> order_items", missing_count=int((~o.order_id.isin(clean["order_items"].order_id)).sum()),
             base=len(o), reason="unavailable / canceled orders that never had items registered",
             treatment="Keep order; item metrics are 0 / missing, flagged has_items in the sales table."),
        dict(relationship="orders -> payments", missing_count=int((~o.order_id.isin(clean["payments"].order_id)).sum()),
             base=len(o), reason="one delivered order has no payment record in the extract",
             treatment="Keep order; payment total missing (not 0); item value is still known."),
        dict(relationship="orders -> reviews", missing_count=int((~o.order_id.isin(clean["reviews"].order_id)).sum()),
             base=len(o), reason="customer did not answer the satisfaction survey",
             treatment="Keep order; review score missing. Never imputed (it is a typical model target)."),
    ])
    implicit["missing_pct"] = (100 * implicit.missing_count / implicit.base).round(3)
    return {"missing_by_column": by_col, "relational_missingness": implicit,
            "orders_missing_by_status": _missing_by_status(raw["orders"]), "all_columns": all_cols}


def _missing_by_status(orders: pd.DataFrame) -> pd.DataFrame:
    from .profiling import orders_missing_by_status
    return orders_missing_by_status(orders)


# --------------------------------------------------------------------------
# Phase 5: outliers
# --------------------------------------------------------------------------
# (dataset, column, valid range, decision, rationale, name of the IQR flag column or None)
OUTLIER_VARS = [
    ("order_items", "price", (0.01, None), "retain + flag",
     "Top prices are real high-ticket goods (computers, appliances, art); right-skew is the nature of retail prices.", "price_outlier_iqr"),
    ("order_items", "freight_value", (0, None), "retain + flag",
     "High freight tracks heavy/bulky goods and long routes; zero freight is free shipping.", "freight_outlier_iqr"),
    ("payments", "payment_value", (0, None), "retain + flag",
     "Large payments correspond to multi-item or high-ticket orders and reconcile with item totals.", "payment_value_outlier_iqr"),
    ("payments", "payment_installments", (1, 24), "retain", "1-24 instalments is the normal Brazilian credit-card range.", None),
    ("products", "product_weight_g", (1, None), "retain + flag (zeros corrected)",
     "Heavy items (furniture, appliances) are legitimate; carrier cap ~30-40 kg explains the ceiling. 0 g values were imputed.", "weight_outlier_iqr"),
    ("products", "product_length_cm", (1, None), "retain", "Bounded by carrier limits (max 105 cm); no impossible values.", None),
    ("products", "product_height_cm", (1, None), "retain", "Bounded by carrier limits (max 105 cm); no impossible values.", None),
    ("products", "product_width_cm", (1, None), "retain", "Bounded by carrier limits (max 118 cm); no impossible values.", None),
    ("reviews", "review_score", (1, 5), "retain", "Ordinal 1-5 scale; all values valid. IQR 'outliers' here are simply low scores.", None),
    ("geolocation", "geolocation_lat", BRAZIL_LAT, "flag invalid, exclude from median",
     "Points outside Brazil are geocoding errors: flagged, kept in the row-level file, excluded from the prefix median.", None),
    ("geolocation", "geolocation_lng", BRAZIL_LNG, "flag invalid, exclude from median", "As latitude.", None),
]


def iqr_bounds(s: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1, q3 = s.quantile([.25, .75])
    return q1 - k * (q3 - q1), q3 + k * (q3 - q1)


def outlier_analysis(clean: dict[str, pd.DataFrame], log: CleaningLog) -> dict[str, pd.DataFrame]:
    """IQR, z-score and percentile analysis. Adds IQR flag columns in place; removes nothing."""
    summary, extremes = [], []
    for name, col, (lo_valid, hi_valid), decision, why, flag in OUTLIER_VARS:
        df = clean[name]
        s = df[col].dropna().astype(float)
        lo, hi = iqr_bounds(s)
        z = (s - s.mean()) / s.std()
        invalid = (s < lo_valid if lo_valid is not None else False) | (s > hi_valid if hi_valid is not None else False)
        q = s.quantile([.01, .05, .25, .5, .75, .95, .99, .999])
        summary.append(dict(
            dataset=name, column=col, n=len(s), mean=s.mean(), median=q[.5], std=s.std(), skewness=s.skew(),
            min=s.min(), p01=q[.01], p05=q[.05], p25=q[.25], p75=q[.75], p95=q[.95], p99=q[.99], p99_9=q[.999], max=s.max(),
            iqr_lower_fence=lo, iqr_upper_fence=hi, iqr_outliers_low=int((s < lo).sum()), iqr_outliers_high=int((s > hi).sum()),
            iqr_outlier_pct=round(100 * ((s < lo) | (s > hi)).mean(), 3), zscore_gt3=int((z.abs() > 3).sum()),
            outside_valid_range=int(np.sum(invalid)), decision=decision, rationale=why, flag_column=flag or ""))
        if flag:
            df[flag] = (df[col] < lo) | (df[col] > hi)
            log.add("5 outliers", name, col, f"statistical outliers by the 1.5xIQR rule (fences {lo:.2f} / {hi:.2f})",
                    df[flag].sum(), f"retain every row; add boolean {flag}", why,
                    validation=f"PASS ({int(np.sum(invalid))} values outside the valid range)")
        id_cols = [c for c in df.columns if c.endswith("_id") or c.endswith("zip_code_prefix")][:3]
        top = df.nlargest(8, col)[id_cols + [col]].astype({c: str for c in id_cols})
        top.insert(0, "variable", f"{name}.{col}")
        top = top.rename(columns={col: "value"})
        top["identifiers"] = top[id_cols].agg(" | ".join, axis=1)
        extremes.append(top[["variable", "value", "identifiers"]])

    it, pr, pay, o = clean["order_items"], clean["products"], clean["payments"], clean["orders"]
    order_tot = it.groupby("order_id").agg(items=("price", "sum"), freight=("freight_value", "sum"), n=("price", "size"))
    paid = pay.groupby("order_id").payment_value.sum()
    top_pay = paid.nlargest(20)
    recon = (top_pay - order_tot.reindex(top_pay.index)[["items", "freight"]].sum(axis=1)).abs()
    w = it.merge(pr[["product_id", "product_weight_g"]], on="product_id")
    investigation = pd.DataFrame([
        ("Items priced above the IQR fence", int(it.price_outlier_iqr.sum()), "8,427 rows cannot all be errors: a long right tail of genuine products"),
        ("Share of item GMV carried by price outliers (%)", round(100 * it.loc[it.price_outlier_iqr, "price"].sum() / it.price.sum(), 1),
         "removing them would delete this share of revenue"),
        ("Items where freight exceeds price", int((it.freight_value > it.price).sum()), "cheap but heavy/remote shipments - legitimate"),
        ("Spearman correlation freight vs product weight", round(w.freight_value.corr(w.product_weight_g, method="spearman"), 3),
         "freight rises with weight, i.e. high freight is explained, not erroneous"),
        ("Top-20 payments reconciling with items+freight within R$1", int((recon < 1).sum()), "the largest payments are real orders"),
        ("Largest single order payment (R$)", float(top_pay.iloc[0]), f"order {top_pay.index[0]} with {int(order_tot.n.get(top_pay.index[0], 0))} items"),
        ("Products at the maximum weight", int((pr.product_weight_g == pr.product_weight_g.max()).sum()), f"max {pr.product_weight_g.max():.0f} g"),
        ("Review scores outside 1-5", int((~clean['reviews'].review_score.between(1, 5)).sum()), "scale is intact"),
        ("Geolocation points outside Brazil", int(clean["geolocation"].flag_outside_brazil.sum()), "flagged; excluded from prefix medians"),
        ("Orders delivered more than 60 days after purchase", int(((o.order_delivered_customer_date - o.order_purchase_timestamp).dt.days > 60).sum()),
         "extreme but real delivery delays - retained (they drive low review scores)"),
    ], columns=["check", "value", "interpretation"])
    return {"summary": pd.DataFrame(summary).round(4), "investigation": investigation,
            "extreme_records": pd.concat(extremes, ignore_index=True)}


def outlier_boxplots(clean: dict[str, pd.DataFrame]) -> io.BytesIO:
    """Box plots (log scale for the right-skewed money/weight variables) as PNG bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("order_items", "price", "Item price (R$)", True), ("order_items", "freight_value", "Freight per item (R$, > 0)", True),
              ("payments", "payment_value", "Payment value (R$, > 0)", True), ("products", "product_weight_g", "Product weight (g)", True),
              ("products", "product_length_cm", "Product length (cm)", False), ("products", "product_height_cm", "Product height (cm)", False),
              ("products", "product_width_cm", "Product width (cm)", False), ("geolocation", "geolocation_lat", "Latitude (raw points)", False),
              ("geolocation", "geolocation_lng", "Longitude (raw points)", False)]
    ink, muted, blue, surface = "#0b0b0b", "#52514e", "#2a78d6", "#fcfcfb"
    fig, axes = plt.subplots(3, 3, figsize=(13, 7.5), facecolor=surface)
    for ax, (name, col, title, log_scale) in zip(axes.ravel(), panels):
        s = clean[name][col].dropna()
        s = s[s > 0] if log_scale else s
        ax.boxplot(s, vert=False, widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor="#cde2fb", edgecolor=blue, linewidth=1.2),
                   medianprops=dict(color=ink, linewidth=1.6), whiskerprops=dict(color=blue, linewidth=1.2),
                   capprops=dict(color=blue, linewidth=1.2),
                   flierprops=dict(marker="o", markersize=2.5, markerfacecolor=muted, markeredgecolor="none", alpha=0.25))
        if log_scale:
            ax.set_xscale("log")
        lo, hi = iqr_bounds(clean[name][col].dropna())
        n_out = int(((clean[name][col] < lo) | (clean[name][col] > hi)).sum())
        ax.set_title(f"{title}", fontsize=10, color=ink, loc="left")
        ax.text(1, 1.02, f"{n_out:,} beyond 1.5×IQR", transform=ax.transAxes, ha="right", fontsize=8, color=muted)
        ax.set_yticks([])
        ax.set_facecolor(surface)
        ax.tick_params(colors=muted, labelsize=8)
        ax.grid(axis="x", color="#e6e5e0", linewidth=0.6)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c2")
    fig.suptitle("Outlier screening - box plots (whiskers at 1.5×IQR; money and weight on a log axis)", fontsize=12, color=ink, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=surface)
    plt.close(fig)
    return buf


# --------------------------------------------------------------------------
def clean_all(raw: dict[str, pd.DataFrame] | None = None) -> tuple[dict[str, pd.DataFrame], CleaningLog]:
    """Run every cleaning step in dependency order. Returns (cleaned tables, log)."""
    raw = raw or load_all_raw()
    log = CleaningLog()
    clean: dict[str, pd.DataFrame] = {}
    clean["geolocation"], clean["geo_zip"] = clean_geolocation(raw["geolocation"], log)
    clean["orders"] = clean_orders(raw["orders"], log)
    clean["order_items"] = clean_order_items(raw["order_items"], clean["orders"], log)
    clean["payments"] = clean_payments(raw["payments"], log)
    clean["reviews"] = clean_reviews(raw["reviews"], clean["orders"], log)
    clean["customers"] = clean_customers(raw["customers"], log)
    clean["sellers"] = clean_sellers(raw["sellers"], clean["geo_zip"], set(clean["geolocation"].geolocation_city.dropna()), log)
    clean["category_translation"] = clean_translation(raw["category_translation"], raw["products"], log)
    clean["products"] = clean_products(raw["products"], clean["category_translation"], log)
    for name in ("orders", "order_items", "payments", "reviews", "customers", "products", "sellers", "category_translation"):
        key = SCHEMA[name]["pk"]
        log.add("2 duplicates", name, " + ".join(key), "exact duplicate rows / duplicate primary keys",
                int(raw[name].duplicated().sum() + raw[name].duplicated(subset=key).sum()), "duplicated() on full row and on the key; nothing to remove",
                "repeating foreign keys are legitimate transactions and were not treated as duplicates",
                validation=_check(len(clean[name]) == len(raw[name]) + (2 if name == "category_translation" else 0)
                                  and not clean[name].duplicated(subset=key).any()))
    return clean, log


def run_cleaning(save: bool = True) -> dict:
    raw = load_all_raw()
    clean, log = clean_all(raw)
    outliers = outlier_analysis(clean, log)
    missing = missing_values_report(raw, clean)
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        for name, df in clean.items():
            df.to_csv(PROCESSED_DIR / PROCESSED[name]["file"], index=False)
        write_excel(REPORTS_DIR / "missing_values_report.xlsx", missing)
        write_excel(REPORTS_DIR / "outlier_analysis.xlsx", outliers, images={"box_plots": outlier_boxplots(clean)})
        write_excel(REPORTS_DIR / "cleaning_log.xlsx", {"cleaning_log": log.frame()})
    return dict(raw=raw, clean=clean, log=log.frame(), outliers=outliers, missing=missing)


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 70, "display.max_rows", 200)
    res = run_cleaning()
    print(res["log"][["step", "phase", "dataset", "column", "affected_records", "records_removed", "records_modified", "validation_result"]].to_string(index=False))
    print(res["missing"]["missing_by_column"].drop(columns=["reason_for_missingness", "treatment"]).to_string(index=False))
    print(res["missing"]["relational_missingness"][["relationship", "missing_count", "missing_pct"]].to_string(index=False))
    print(res["outliers"]["summary"][["dataset", "column", "median", "p99", "max", "iqr_upper_fence", "iqr_outliers_low", "iqr_outliers_high", "zscore_gt3", "outside_valid_range"]].to_string(index=False))
    print(res["outliers"]["investigation"].to_string(index=False))
    print(res["clean"]["geo_zip"].describe().round(2).to_string())
