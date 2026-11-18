"""Offline tests for the RAG chatbot.  Run:  python -X utf8 chatbot/test_chatbot.py

Checks kind, sanity numbers, record tables, read-only SQL, robustness against odd input, untouched table
values, and that the database file is byte-for-byte unchanged afterwards.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import time
from pathlib import Path

os.environ["CHATBOT_OFFLINE"] = "1"                       # tests must pass without an API key / LLM
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from chatbot.RAG import Chatbot  # noqa: E402

DB = ROOT / "database" / "ecommerce.db"
TOTAL, JULY, NET_ALL, RETURNING = "R$ 16,008,872.12", "R$ 1,066,540.75", "R$ 15,739,137.01", "2,997"
CUSTOMER = "0000366f3b9a7992bf8c76cfdf3221e2"
H, P, E, M, U = "historical", "prediction", "explanation", "mixed", "unavailable"

# (question, expected kind(s), strings that must appear in the answer, record tables required)
CASES = [
    # total revenue last month
    ("What was the total revenue last month?", H, [JULY, "July 2018"], True),
    ("How much revenue did we make last month?", H, [JULY, "July 2018"], True),
    ("Show me sales for the previous month", H, [JULY], True),
    ("Revenue in July 2018", H, [JULY], True),
    ("What was revenue in 2018-07?", H, [JULY], True),
    ("What is the total revenue?", H, [TOTAL], True),
    ("Total sales all time", H, [TOTAL], True),
    ("How many orders were there in 2017?", H, ["orders"], True),
    ("Revenue by month in 2018", H, [], True),
    # top products
    ("Which products generated the highest sales?", H, ["not in the dataset"], True),
    ("What are the best-selling products?", H, [], True),
    ("Top 5 products by revenue in 2018", H, [], True),
    ("Show the top 10 items by units sold", H, [], True),
    # customers not purchased recently
    ("Which customers have not purchased recently?", H, ["180 days"], True),
    ("Which customers haven't bought anything in the last 90 days?", H, ["90 days"], True),
    ("List inactive customers", H, [], True),
    # predicted demand
    ("What is the predicted demand for toys?", P, ["toys", "forecast"], True),
    ("Forecast demand for health and beauty next 2 weeks", P, ["health_beauty", "forecast"], True),
    ("What is the expected demand for bed_bath_table next month?", P, ["bed_bath_table"], True),
    ("Predicted demand for helth beauty", P, ["health_beauty"], True),
    ("What will demand look like next month?", P, ["forecast"], True),
    ("What is the predicted demand for flowers?", H, ["No demand forecast exists"], True),
    ("Demand forecast for product aca2eb7d00ea1a7b8ebd4e68314663af", H, ["No demand forecast exists for single products"], True),
    # expected revenue
    ("What is the expected revenue for the next month?", P, ["Forecast", "2018-08-20"], True),
    ("Revenue forecast for the next 2 weeks", P, ["Forecast", "next 14 days"], True),
    ("How much revenue will we make next quarter?", P, ["Forecast"], True),
    ("Predict net revenue for the next 30 days", P, ["Forecast", "net revenue"], True),
    ("What is the expected revenue in September 2018?", P, ["Forecast"], True),
    ("What will be the revenue tomorrow?", P, ["Forecast", "next day", "2018-08-20 to 2018-08-20"], True),
    ("expected revenue in 2 weeks", P, ["Forecast", "next 14 days"], True),
    ("revenue 20 days ahead", P, ["Forecast", "next 20 days"], True),
    ("forecast for next 90 days", P, ["Forecast", "next 90 days", "No subject was named"], True),
    ("predict revenue for Q4 2018", P, ["Forecast", "2018-10-01 to 2018-11-17", "only the part"], True),
    ("revenue forecast for next year", P, ["Forecast", "limited to 90 days"], True),
    # declining demand
    ("Which products have declining demand?", P, ["declining"], True),
    ("Which categories are declining?", P, ["declining"], True),
    ("Which categories have growing demand?", P, ["growing"], True),
    # returning customers
    ("How many customers are returning customers?", H, [RETURNING], True),
    ("How many repeat customers do we have?", H, [RETURNING], True),
    ("How many customers ordered more than once?", H, [RETURNING], True),
    # churn
    ("Which customers have a high churn probability?", P, ["synthetic"], True),
    ("Show customers at risk of churning", P, ["synthetic"], True),
    ("Top 5 customers most likely to churn", P, ["synthetic"], True),
    ("What is the churn rate?", P, ["synthetic"], True),
    (f"What is the churn risk of customer {CUSTOMER}?", M, ["0.5021", "synthetic", "model estimate"], True),
    (f"churn probability for {CUSTOMER}", M, ["0.5021"], True),
    ("What is the churn risk of customer ffffffffffffffffffffffffffffffff?", U, ["not a customer"], False),
    # net revenue
    ("What was net revenue from 2018-01 to 2018-06? How much was lost to cancellations?", H, ["R$ 6,557,889.21", "R$ 27,229.70"], True),
    ("What was net revenue in March 2018?", H, ["March 2018", "Net revenue"], True),
    ("What is the net revenue?", H, [NET_ALL], True),
    ("How much was lost to cancellations?", H, ["R$ 143,255.60"], True),
    ("Net revenue after refunds and returns in 2018", H, ["not available"], True),
    # categories / states
    ("Top 5 categories by revenue in 2018", H, ["health_beauty"], True),
    ("Which categories sold the most units?", H, ["bed_bath_table"], True),
    ("Revenue by state", H, ["SP", "R$ 5,998,226.96"], True),
    ("Which state generates the most sales?", H, ["SP"], True),
    ("What was revenue in Sao Paulo in 2017?", H, ["SP"], True),
    # explanations
    ("How does the ensemble work?", E, ["0.8", "weighted average"], False),
    ("Explain the weighted ensemble", E, ["validation"], False),
    ("What is churn?", E, ["180 days"], False),
    ("What does churn mean?", E, ["180 days"], False),
    ("How is net revenue calculated?", E, ["cancelled", "unavailable"], False),
    ("Why are synthetic orders used?", E, ["synthetic"], False),
    ("What are the data limitations?", E, ["2018-08-19"], False),
    # accuracy
    ("How accurate is the revenue model?", E, ["MAPE"], True),
    ("How good is the demand model?", E, ["WAPE"], True),
    ("What is the ROC AUC of the churn model?", E, ["ROC-AUC"], True),
    # unavailable information
    ("What were the discounts or refunds?", U, ["discount", "refund"], False),
    ("How much profit did we make in 2017?", U, ["Profit"], False),
    ("What is our margin?", U, ["margin"], False),
    ("What are the names of the top products?", U, ["Product names"], False),
    ("What are the customer names?", U, ["names"], False),
    ("How many returns did we have?", U, ["returns"], False),
    ("What was the cost of goods sold?", U, ["cost"], False),
    # overall analysis
    ("Give me an overall business analysis", M, [TOTAL, NET_ALL, "Forecast", "Recommendations"], True),
    ("How is the business doing?", M, [TOTAL], True),
    # other historical questions
    ("How many customers do we have?", H, ["96,096"], True),
    ("What was the average order value in 2018?", H, ["average order value"], True),
    ("What was revenue in August 2018?", H, ["incomplete"], True),
    ("What is the average review score?", H, ["4.09"], True),
    ("How do customers pay? Show payment methods", H, ["credit_card"], True),
    ("What is the average delivery time?", H, ["days"], True),
    ("Who are the top sellers?", H, ["sellers"], True),
    ("Who are our best customers?", H, ["Top 10 customers"], True),
    ("Orders by status", H, ["delivered"], True),
    ("Tell me about product aca2eb7d00ea1a7b8ebd4e68314663af", H, ["furniture_decor"], True),
    ("Sales of toys in 2017", H, ["Actual sales of toys"], True),
    # gibberish / off-topic
    ("asdkjh qwe zzz", U, ["You can ask"], False),
    ("Who won the world cup in 2018?", U, ["You can ask"], False),
    ("What's the weather like today?", U, ["You can ask"], False),
    ("Tell me a joke", U, ["You can ask"], False),
    ("What is the capital of France?", U, ["You can ask"], False),
    ("Revenue in 2019", U, ["no orders"], False),
]

ROBUSTNESS = ["", "   ", None, 12345, "?", "!!!", "a" * 20000, "revenue " * 3000, "'; DROP TABLE orders; --",
              "DELETE FROM orders", "SELECT * FROM orders; DROP TABLE customers", "1' OR '1'='1", "Robert'); DROP TABLE students;--",
              "revenue in 2018'; DELETE FROM payments; --", "\x00\x01\x02", "😀 revenue 😀", "revenue in 2018-13", "top 0 products",
              "top 99999 products", "next 1000 days revenue forecast", "churn risk of customer 'OR 1=1", "demand for ' OR 1=1 --",
              "what was revenue between december 2017 and february 2018", "Revenue for Q1 2018", "orders yesterday"]

FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex)\b", re.I)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_structure(resp, errors: list, label: str) -> None:
    if not isinstance(resp, dict) or set(resp) != {"answer", "kind", "tables", "sql", "sources"}:
        errors.append(f"{label}: wrong response structure")
        return
    if resp["kind"] not in {H, P, E, M, U} or not isinstance(resp["answer"], str) or not resp["answer"]:
        errors.append(f"{label}: bad kind/answer")
    for t in resp["tables"]:
        if not isinstance(t.get("title"), str) or not isinstance(t.get("data"), pd.DataFrame) or t["data"].empty:
            errors.append(f"{label}: bad table entry")
    for s in resp["sql"]:
        if not re.match(r"^\s*(select|with)\b", s, re.I) or FORBIDDEN.search(re.sub(r"'[^']*'", "''", s)):
            errors.append(f"{label}: non-SELECT SQL: {s[:80]}")
    if not all(isinstance(s, str) for s in resp["sources"]):
        errors.append(f"{label}: bad sources")


def main() -> int:
    before = sha256(DB)
    mtime = DB.stat().st_mtime
    bot = Chatbot()
    passed = failed = 0
    t0 = time.time()

    for q, kinds, must, need_tables in CASES:
        errors = []
        try:
            r = bot.ask(q)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL (raised {type(exc).__name__}: {exc}) {q}")
            failed += 1
            continue
        check_structure(r, errors, q)
        kinds = kinds if isinstance(kinds, (list, tuple)) else [kinds]
        if r["kind"] not in kinds:
            errors.append(f"kind {r['kind']} not in {kinds}")
        for s in must:
            if s.lower() not in r["answer"].lower():
                errors.append(f"missing '{s}'")
        if need_tables and not r["tables"]:
            errors.append("no tables")
        if r["kind"] == P and not re.search(r"forecast|model estimate", r["answer"], re.I):
            errors.append("prediction not labelled as forecast/model estimate")
        if "churn" in q.lower() and r["kind"] in (P, M) and "synthetic" not in r["answer"]:
            errors.append("churn answer does not mention synthetic orders")
        if errors:
            failed += 1
            print(f"FAIL {q!r}: {'; '.join(errors)}\n     -> [{r['kind']}] {r['answer'][:300]!r}")
        else:
            passed += 1

    for q in ROBUSTNESS:
        errors = []
        try:
            r = bot.ask(q)
            check_structure(r, errors, repr(q)[:40])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"raised {type(exc).__name__}: {exc}")
        if errors:
            failed += 1
            print(f"FAIL robustness {repr(q)[:60]}: {'; '.join(errors)}")
        else:
            passed += 1

    # retrieved records are returned unmodified
    checks = [
        ("Which products generated the highest sales?", "SELECT * FROM v_product_sales ORDER BY items_value DESC LIMIT 10"),
        ("Revenue by state", "SELECT delivery_state AS state, COUNT(*) AS orders, COUNT(DISTINCT customer_unique_id) AS customers, "
                             "ROUND(SUM(payment_total), 2) AS revenue, ROUND(AVG(payment_total), 2) AS avg_order_value FROM v_sales "
                             "GROUP BY delivery_state ORDER BY revenue DESC"),
        (f"churn risk of customer {CUSTOMER}", f"SELECT * FROM churn_predictions WHERE customer_unique_id = '{CUSTOMER}'"),
    ]
    for q, sql in checks:
        r = bot.ask(q)
        expected = bot.c.query(sql)
        ok = any(t["data"].reset_index(drop=True).equals(expected) for t in r["tables"])
        passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
        if not ok:
            print(f"FAIL table values changed for {q!r}")

    after = sha256(DB)
    if after == before:
        passed += 1
    else:
        failed += 1
        print("FAIL database file changed")
    print(f"(database mtime unchanged: {DB.stat().st_mtime == mtime})")
    print(f"\n{passed} passed, {failed} failed ({len(CASES)} questions, {len(ROBUSTNESS)} robustness inputs, "
          f"{len(checks)} table checks, 1 database check) in {time.time() - t0:.0f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
