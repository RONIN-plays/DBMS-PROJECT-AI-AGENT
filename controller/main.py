"""Central controller - the single entry point for the dashboard and the chatbot.

It decides which agent a request needs, validates the inputs, loads the trained ensemble,
reads the database, and returns the result. It coordinates the four agents; it never blends
their different targets into one number.

    feature          agent                    main parameters
    ---------------  -----------------------  ---------------------------------------------
    overview         SQL only                 -
    revenue          models/revenue.py        horizon_days (1-90)
    net_revenue      models/netrevenue.py     start_month, end_month ('YYYY-MM'), horizon_days
    demand           models/demand.py         weeks (1-12), category (optional)
    churn            models/churn.py          customer_unique_id (optional), top_n
    full_analysis    all four                 -

Python:   from controller.main import Controller
          c = Controller()
          c.run("revenue", horizon_days=30)            # rich result (DataFrames) or raises
          c.handle({"feature": "revenue", "horizon_days": 30})   # JSON-safe dict, never raises
          c.handle("which products will be in demand next month?")  # free text is routed by keywords
Shell:    python controller/main.py revenue --horizon_days 30
          python controller/main.py train             # (re)train all four agents
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import churn, demand, netrevenue, revenue  # noqa: E402
from models.ensemble import DATA_END, DATA_START, DB_PATH  # noqa: E402

FEATURES = ["overview", "revenue", "net_revenue", "demand", "churn", "full_analysis"]
AGENT_MODULES = {"churn": churn, "revenue": revenue, "netrevenue": netrevenue, "demand": demand}

# free-text routing: first matching rule wins (most specific first)
KEYWORDS = [
    ("full_analysis", r"overall|full analysis|business analysis|everything|all (four )?agents|summary of the business"),
    ("net_revenue", r"net revenue|net sales|deduction|leakage|cancel|refund|return(s|ed)? (value|amount)|discount|unavailable"),
    ("churn", r"churn|retention|retain|at.risk|stop(ped)? (buying|purchasing)|inactive|lose customers|loyal"),
    ("demand", r"demand|inventory|stock|categor(y|ies)|units|best.sell|top.sell|product"),
    ("revenue", r"revenue|sales forecast|turnover|income|forecast|predict"),
    ("overview", r"overview|kpi|dashboard|how (is|are) (the )?business|total (orders|customers)"),
]
MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
FORBIDDEN_SQL = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex)\b", re.I)


class RequestError(ValueError):
    """The request itself is wrong (unknown feature, bad parameter). Safe to show to the user."""


def identify_feature(text: str) -> str:
    """Map a free-text request to a feature name using keyword rules."""
    low = text.lower()
    for feature, pattern in KEYWORDS:
        if re.search(pattern, low):
            return feature
    raise RequestError(f"Could not tell which analysis is wanted. Choose one of: {', '.join(FEATURES)}.")


def to_jsonable(obj):
    """Convert DataFrames, Series, numpy and pandas scalars into plain JSON-safe Python."""
    if isinstance(obj, pd.DataFrame):
        df = obj.reset_index() if obj.index.name or isinstance(obj.index, pd.DatetimeIndex) else obj
        return [to_jsonable(r) for r in df.to_dict("records")]
    if isinstance(obj, pd.Series):
        return [{"index": to_jsonable(i), "value": to_jsonable(v)} for i, v in obj.items()]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, pd.Period)):
        return str(obj.date()) if isinstance(obj, pd.Timestamp) and obj == obj.normalize() else str(obj)
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if obj is pd.NaT or obj is pd.NA:
        return None
    return obj


class Controller:
    """Coordinates the database and the four agents."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}. Build it with: python database/build_db.py")

    # ------------------------------------------------------------ database
    def _connect(self) -> sqlite3.Connection:
        """Read-only connection: nothing reached through the controller can change the data."""
        return sqlite3.connect(f"file:{self.db_path.as_posix()}?mode=ro", uri=True)

    def query(self, sql: str, params: tuple | dict = ()) -> pd.DataFrame:
        """Run ONE read-only SELECT (or WITH ... SELECT) and return the rows."""
        stripped = sql.strip().rstrip(";")
        if ";" in stripped or not re.match(r"^(select|with)\b", stripped, re.I) or FORBIDDEN_SQL.search(stripped):
            raise RequestError("Only a single read-only SELECT statement is allowed.")
        con = self._connect()
        try:
            return pd.read_sql_query(stripped, con, params=params)
        except (sqlite3.Error, pd.errors.DatabaseError) as exc:
            raise RequestError(f"SQL error: {exc}") from exc
        finally:
            con.close()

    def schema(self) -> dict[str, list[str]]:
        """Tables and views with their columns (used by the chatbot to ground SQL retrieval)."""
        names = self.query("SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' ORDER BY name")
        return {n: self.query(f"SELECT name FROM pragma_table_info('{n}')").name.tolist() for n in names.name}

    # ------------------------------------------------------------ features
    def overview(self) -> dict:
        """Headline KPIs. Everything here is a historical fact from REAL orders, except the churn block."""
        k = self.query("""
            SELECT COUNT(*) AS total_orders, COUNT(DISTINCT customer_unique_id) AS total_customers,
                   SUM(order_status = 'delivered') AS delivered_orders, SUM(n_items) AS total_units_sold,
                   ROUND(SUM(payment_total), 2) AS total_revenue,
                   ROUND(SUM(CASE WHEN order_status NOT IN ('canceled', 'unavailable') THEN payment_total ELSE 0 END), 2) AS net_revenue,
                   ROUND(SUM(items_value), 2) AS items_value, ROUND(SUM(freight_value), 2) AS freight_value,
                   ROUND(AVG(payment_total), 2) AS avg_order_value, ROUND(AVG(review_score), 2) AS avg_review_score,
                   MIN(purchase_date) AS first_order_date, MAX(purchase_date) AS last_order_date
            FROM v_sales""").iloc[0].to_dict()
        returning = int(self.query("SELECT COUNT(*) AS n FROM v_customer_summary WHERE is_repeat_customer").n[0])
        risk = self.query("""SELECT risk_category, COUNT(*) AS customers FROM churn_predictions GROUP BY risk_category
                             ORDER BY CASE risk_category WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END""")
        return dict(
            kpis={**k, "returning_customers": returning,
                  "returning_customer_pct": round(100 * returning / k["total_customers"], 2)},
            churn_risk=dict(by_category=risk, customers_scored=int(risk.customers.sum()),
                            note="Model output; trained on real plus synthetic repeat orders."),
            monthly_revenue=self.query("SELECT * FROM v_monthly_revenue ORDER BY purchase_month"),
            top_categories=self.query("""SELECT category, SUM(units) AS units, ROUND(SUM(items_value), 2) AS items_value
                                         FROM v_category_weekly_demand GROUP BY category ORDER BY units DESC LIMIT 10"""),
            top_products=self.query("SELECT * FROM v_product_sales ORDER BY units DESC LIMIT 10"),
            order_status=self.query("SELECT order_status, COUNT(*) AS orders FROM v_sales GROUP BY order_status ORDER BY orders DESC"),
            data_window=dict(model_training_start=DATA_START, model_training_end=DATA_END,
                             note="Orders after 2018-08-19 are incomplete in the source extract."))

    def revenue(self, horizon_days: int = 30) -> dict:
        return revenue.predict(self._int(horizon_days, "horizon_days", 1, 90))

    def net_revenue(self, start_month: str | None = None, end_month: str | None = None, horizon_days: int = 30) -> dict:
        for name, value in (("start_month", start_month), ("end_month", end_month)):
            if value is not None and not MONTH.match(str(value)):
                raise RequestError(f"{name} must look like 'YYYY-MM' (got '{value}').")
        if start_month and end_month and start_month > end_month:
            raise RequestError("start_month must not be after end_month.")
        result = netrevenue.analyse(start_month, end_month)                       # exact calculation (SQL)
        result["forecast"] = netrevenue.predict(self._int(horizon_days, "horizon_days", 1, 90))   # model prediction
        return result

    def demand(self, weeks: int = 4, category: str | None = None) -> dict:
        return demand.predict(self._int(weeks, "weeks", 1, 12), category or None)

    def churn(self, customer_unique_id: str | None = None, top_n: int = 20) -> dict:
        return churn.predict(customer_unique_id or None, self._int(top_n, "top_n", 1, 500))

    def full_analysis(self) -> dict:
        """Run every agent and present the outputs side by side, plus cross-agent recommendations."""
        out = dict(overview=self.overview(), revenue=self.revenue(), net_revenue=self.net_revenue(),
                   demand=self.demand(), churn=self.churn())
        out["recommendations"] = self.recommendations(out)
        return out

    @staticmethod
    def recommendations(r: dict) -> list[str]:
        """Plain-language actions derived from the agents' outputs (rules, not another model)."""
        rev, net, dem, chn = r["revenue"], r["net_revenue"], r["demand"], r["churn"]
        recs = [f"Revenue: the next {rev['horizon_days']} days are forecast at R${rev['forecast_total']:,.0f} "
                f"({rev['growth_vs_previous_period_pct']:+.1f}%, {rev['trend']}). "
                + {"growing": "Make sure stock and logistics capacity can absorb the growth.",
                   "declining": "Plan promotions or acquisition spend to counter the decline.",
                   "stable": "No capacity change needed; growth has to come from retention or new categories."}[rev["trend"]]]
        pay = net["leakage_by_payment"]
        worst = pay.iloc[0]
        recs.append(f"Net revenue: R${net['total_deductions']:,.0f} ({100 - net['net_revenue_pct_of_gross']:.2f}% of billed) was lost to "
                    f"cancelled and unavailable orders. '{worst.segment}' orders lose the most ({worst.lost_pct_of_billed:.1f}%); "
                    "review stock checks at checkout to cut 'unavailable' orders.")
        if dem["increasing"]:
            recs.append(f"Demand: raise stock for {', '.join(dem['increasing'][:4])} (forecast up more than 10%).")
        if dem["decreasing"]:
            recs.append(f"Demand: trim the next orders for {', '.join(dem['decreasing'][:4])} (forecast down more than 10%).")
        att = chn["customers_needing_retention_attention"]
        recs.append(f"Churn: {chn['predicted_to_churn']:,} of {chn['customers_scored']:,} customers are predicted to churn. "
                    f"Start with the {len(att)} highest-spending repeat customers at risk "
                    f"(R${att.total_paid.sum():,.0f} of past spend); the main driver is {chn['factors'].description.iloc[0]}.")
        return recs

    # ------------------------------------------------------------ support
    def model_evaluation(self, agent: str | None = None) -> pd.DataFrame:
        """Validation and test metrics of every model and of the ensemble, with the ensemble weights."""
        if agent is not None and agent not in AGENT_MODULES:
            raise RequestError(f"agent must be one of: {', '.join(AGENT_MODULES)}")
        df = self.query("SELECT agent, model, weight, split, metric, value FROM model_evaluation")
        if agent:
            df = df[df.agent == agent]
        wide = df.pivot_table(index=["agent", "split", "model"], columns="metric", values="value").reset_index()
        weights = df.drop_duplicates(["agent", "model"])[["agent", "model", "weight"]]
        wide = wide.merge(weights, on=["agent", "model"], how="left")
        wide.columns.name = None
        return wide[["agent", "split", "model", "weight"] + [c for c in wide.columns if c not in ("agent", "split", "model", "weight")]]

    def train_all(self) -> dict:
        """(Re)train the four agents in dependency order. Needs the database; rewrites the .pkl files."""
        return {name: {"weights": module.train()["weights"]} for name, module in AGENT_MODULES.items()}

    @staticmethod
    def _int(value, name: str, lo: int, hi: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise RequestError(f"{name} must be a whole number between {lo} and {hi}.") from None
        if not lo <= number <= hi:
            raise RequestError(f"{name} must be between {lo} and {hi} (got {number}).")
        return number

    # ------------------------------------------------------------ dispatch
    def run(self, feature: str, **params) -> dict:
        """Execute one feature and return the rich result. Raises RequestError on bad input."""
        if feature not in FEATURES:
            raise RequestError(f"Unknown feature '{feature}'. Choose one of: {', '.join(FEATURES)}.")
        method = getattr(self, feature)
        try:
            return method(**params)
        except TypeError as exc:
            if "unexpected keyword argument" in str(exc):
                raise RequestError(f"Parameter not accepted by '{feature}': {exc}") from None
            raise
        except ValueError as exc:                      # validation errors raised inside the agents
            raise RequestError(str(exc)) from None

    def handle(self, request: dict | str) -> dict:
        """UI entry point. Accepts {'feature': ..., **params} or free text; always returns a JSON-safe dict."""
        try:
            if isinstance(request, str):
                request = {"feature": identify_feature(request)}
            if not isinstance(request, dict) or "feature" not in request:
                raise RequestError("Request must be text or a dict with a 'feature' key.")
            params = {k: v for k, v in request.items() if k != "feature"}
            return {"status": "ok", "feature": request["feature"], "result": to_jsonable(self.run(request["feature"], **params))}
        except RequestError as exc:
            return {"status": "error", "error_type": "invalid_request", "message": str(exc)}
        except FileNotFoundError as exc:                # database or a trained model is missing
            return {"status": "error", "error_type": "not_ready", "message": str(exc)}
        except Exception as exc:                        # noqa: BLE001 - the UI must never crash on an agent failure
            return {"status": "error", "error_type": "internal", "message": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser(description="E-commerce BI controller")
    parser.add_argument("feature", choices=FEATURES + ["train"])
    parser.add_argument("--horizon_days", type=int)
    parser.add_argument("--weeks", type=int)
    parser.add_argument("--category")
    parser.add_argument("--customer_unique_id")
    parser.add_argument("--top_n", type=int)
    parser.add_argument("--start_month")
    parser.add_argument("--end_month")
    args = parser.parse_args()
    controller = Controller()
    if args.feature == "train":
        print(json.dumps(controller.train_all(), indent=2))
        return
    params = {k: v for k, v in vars(args).items() if k != "feature" and v is not None}
    print(json.dumps(controller.handle({"feature": args.feature, **params}), indent=2, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
