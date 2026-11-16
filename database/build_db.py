"""Build database/ecommerce.db from the cleaned (and synthetic) CSV files using db.sql.

Steps
  1. create the schema                       (db.sql, section "schema")
  2. bulk-load the CSV files into stg_* staging tables
  3. fill the normalised tables              (db.sql, section "transform")
  4. drop the staging tables, create views   (db.sql, section "views")
  5. create the empty prediction tables       (db.sql, section "predictions"; filled by models/*.py)
  6. run the validation queries and print    (db.sql, section "validation")

Run:  python database/build_db.py
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CLEAN_DIR = ROOT / "dataset" / "clean dataset"
SYNTHETIC_DIR = ROOT / "dataset" / "synthetic dataset"
DB_PATH = HERE / "ecommerce.db"
SQL_PATH = HERE / "db.sql"

# staging table -> (cleaned file, synthetic file or None)
SOURCES = {
    "stg_orders": ("olist_orders_clean.csv", "synthetic_orders.csv"),
    "stg_customers": ("olist_customers_clean.csv", "synthetic_customers.csv"),
    "stg_order_items": ("olist_order_items_clean.csv", "synthetic_order_items.csv"),
    "stg_payments": ("olist_order_payments_clean.csv", "synthetic_payments.csv"),
    "stg_reviews": ("olist_order_reviews_clean.csv", "synthetic_reviews.csv"),
    "stg_products": ("olist_products_clean.csv", None),
    "stg_sellers": ("olist_sellers_clean.csv", None),
    "stg_geo_zip": ("geolocation_zip_reference.csv", None),
}


def read_csv(path: Path) -> pd.DataFrame:
    """IDs and postal prefixes stay text (leading zeros); timestamps stay ISO text."""
    header = pd.read_csv(path, nrows=0).columns
    text = [c for c in header if (c.endswith("_id") and c != "order_item_id") or "zip_code_prefix" in c
            or c.endswith(("_date", "_timestamp", "_at"))]
    return pd.read_csv(path, dtype={c: str for c in text})


def sections(sql: str) -> dict[str, str]:
    parts = re.split(r"^-- @@SECTION: (\w+)\s*$", sql, flags=re.M)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}


def load_staging(con: sqlite3.Connection) -> None:
    for table, (clean_file, synthetic_file) in SOURCES.items():
        df = read_csv(CLEAN_DIR / clean_file)
        if synthetic_file:
            df["is_synthetic"] = False
            syn = read_csv(SYNTHETIC_DIR / synthetic_file)
            df = pd.concat([df, syn[[c for c in df.columns if c in syn.columns]]], ignore_index=True)
        df.to_sql(table, con, index=False, chunksize=50_000)
        print(f"  staged {table:<16} {len(df):>8,} rows")


def run_validation(con: sqlite3.Connection, sql: str) -> None:
    for block in sql.split(";"):
        lines = [ln for ln in block.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        title = lines[0].lstrip("- ").strip() if lines[0].startswith("--") else "query"
        result = pd.read_sql_query("\n".join(lines), con)
        print(f"\n-- {title}")
        print(result.to_string(index=False) if len(result) else "(no rows)")


def build() -> None:
    sec = sections(SQL_PATH.read_text(encoding="utf-8"))
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)
    try:
        con.executescript(sec["schema"])
        print("schema created; loading staging tables")
        load_staging(con)
        con.execute("PRAGMA foreign_keys = ON")
        con.executescript("BEGIN;" + sec["transform"] + "COMMIT;")
        for table in SOURCES:
            con.execute(f"DROP TABLE {table}")
        con.executescript(sec["views"])
        con.executescript(sec["predictions"])
        con.commit()
        con.execute("VACUUM")
        print("tables filled, staging dropped, views created")
        run_validation(con, sec["validation"])
    finally:
        con.close()
    print(f"\ndatabase written: {DB_PATH} ({DB_PATH.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    pd.set_option("display.width", 220, "display.max_columns", 30)
    build()
