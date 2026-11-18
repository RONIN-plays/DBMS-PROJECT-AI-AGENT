"""RAG business chatbot for the Olist e-commerce BI project.

Hybrid retrieval, then generation:
  1. SQL retrieval      exact numbers and records, always through Controller.query (read-only, one SELECT)
  2. Agent retrieval    forecasts and churn scores, through Controller.run(...) or the prediction tables
  3. Knowledge retrieval definitions and explanations: TF-IDF over chatbot/knowledge/*.md
  4. Generation         offline (default): an intent router with parameter extraction picks SQL templates
                        and/or agent calls, and the answer is composed only from the retrieved values.
                        Optional LLM mode (ANTHROPIC_API_KEY set and `anthropic` importable, and
                        CHATBOT_OFFLINE not set): Claude translates questions the router cannot match into
                        ONE SELECT (run through Controller.query) and words answers from the retrieved rows;
                        any number it writes that is not in the retrieved context makes the bot fall back
                        to the offline answer.

Python:  from chatbot.RAG import Chatbot
         Chatbot().ask("What was the total revenue last month?")
         -> {"answer", "kind", "tables", "sql", "sources"}
Shell:   python chatbot/RAG.py "question"      (no argument: interactive loop)
"""
from __future__ import annotations

import difflib
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controller.main import Controller, RequestError  # noqa: E402
from models.ensemble import DATA_END, DATA_START  # noqa: E402  ('2017-01-09', '2018-08-19')

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"

# Time frame of the extract (see knowledge/data_limitations.md)
FIRST_DAY, LAST_DAY = "2016-09-04", "2018-10-17"        # first / last order in the data
LAST_COMPLETE_MONTH = "2018-07"                         # August 2018 is complete only to the 19th
FORECAST_START = "2018-08-20"                           # first forecast day of every agent
FORECAST_LAST = "2018-11-17"                            # last day a 90-day forecast reaches
MAX_ROWS = 500

LLM_MODEL = "claude-opus-5-5"

EXAMPLES = [
    "What was the total revenue last month?",
    "What was net revenue from 2018-01 to 2018-06? How much was lost to cancellations?",
    "Top 5 categories by revenue in 2018",
    "Which products generated the highest sales?",
    "Revenue by state",
    "How many customers are returning customers?",
    "Which customers have not purchased recently?",
    "What is the expected revenue for the next month?",
    "What is the predicted demand for toys?",
    "Which categories have declining demand?",
    "Which customers have a high churn probability?",
    "What is the churn risk of customer <32-character id>?",
    "How does the ensemble work? / What is churn? / How is net revenue calculated?",
    "How accurate is the revenue model?",
    "Give me an overall business analysis",
]

# --------------------------------------------------------------------------- vocabulary
WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
           "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "fifty": 50, "hundred": 100}
NUM = r"(\d+|" + "|".join(WORDNUM) + r")"
MONTH_NAMES = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
MONTH_RE = re.compile(r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
                      r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b(?:[\s,]+(?:of\s+)?(20\d\d))?")
STATES = {"AC": "acre", "AL": "alagoas", "AP": "amapa", "AM": "amazonas", "BA": "bahia", "CE": "ceara", "DF": "distrito federal",
          "ES": "espirito santo", "GO": "goias", "MA": "maranhao", "MT": "mato grosso", "MS": "mato grosso do sul",
          "MG": "minas gerais", "PA": "para", "PB": "paraiba", "PR": "parana", "PE": "pernambuco", "PI": "piaui",
          "RJ": "rio de janeiro", "RN": "rio grande do norte", "RS": "rio grande do sul", "RO": "rondonia", "RR": "roraima",
          "SC": "santa catarina", "SP": "sao paulo", "SE": "sergipe", "TO": "tocantins"}
CATEGORY_ALIASES = {
    "health and beauty": "health_beauty", "beauty": "health_beauty", "cosmetics": "health_beauty",
    "toy": "toys", "furniture": "furniture_decor", "home decor": "furniture_decor", "decor": "furniture_decor",
    "bed and bath": "bed_bath_table", "bed bath": "bed_bath_table", "bedding": "bed_bath_table",
    "sports": "sports_leisure", "sport": "sports_leisure", "computer accessories": "computers_accessories",
    "computers and accessories": "computers_accessories", "phones": "telephony", "mobile phones": "telephony",
    "watches": "watches_gifts", "watch": "watches_gifts", "gifts": "watches_gifts", "garden": "garden_tools",
    "cars": "auto", "car": "auto", "automotive": "auto", "perfume": "perfumery", "perfumes": "perfumery",
    "pets": "pet_shop", "pet": "pet_shop", "pet products": "pet_shop", "video games": "consoles_games",
    "games": "consoles_games", "consoles": "consoles_games", "household": "housewares", "bags": "fashion_bags_accessories",
    "books": "books_general_interest", "electronic": "electronics", "baby products": "baby",
}
FUZZY_STOP = {"revenue", "demand", "sales", "category", "categories", "product", "products", "customer", "customers",
              "predicted", "forecast", "expected", "orders", "states", "monthly", "weekly", "highest", "lowest", "decline",
              "declining", "increase", "increasing", "units", "total", "which", "what", "there", "their", "about", "selling"}

REVENUE_W = r"\b(revenue|revenues|sales|turnover|income|earn\w*|made|make|billed|billing|gmv|takings|money)\b"
PRED_W = r"\b(predict\w*|forecast\w*|expect\w*|project(ed|ion)s?|future|next|upcoming|coming|will|going to|anticipat\w*|estimate[sd]?|outlook|ahead|from now)\b"
PAST_W = r"\b(was|were|did|sold|historical\w*|history|past|actual\w*|so far|last (month|year|quarter|\d+))\b"
DEMAND_W = r"\b(demand|units?|stock|inventory|restock\w*|replenish\w*)\b"
CUSTOMER_W = r"\b(customers?|clients?|buyers?|shoppers?)\b"
SQL_ATTACK = re.compile(r"\b(drop|delete|insert|update|alter|truncate|create|attach|pragma)\s+(table|from|into|database|view|index|\w+\s+set)\b|;\s*--|\bunion\s+select\b", re.I)

UNAVAILABLE = [
    ("discounts", r"\bdiscount\w*|\bmarkdowns?\b|\bpromo(tional)? codes?\b",
     "The dataset contains no discount information (no list prices and no discount fields), so discounts cannot be reported. "
     "Voucher payments do exist, but a voucher is a payment method, not a recorded discount."),
    ("returns", r"\breturns\b|\breturned (items?|products?|goods|orders?)\b|\bproduct returns?\b|\breturn rates?\b",
     "The dataset does not record product returns, so returns cannot be reported. The closest available measure is the value of "
     "cancelled and unavailable orders (ask: \"How much was lost to cancellations?\")."),
    ("refunds", r"\brefund\w*|\bchargebacks?\b",
     "The dataset does not record refunds, so refunds cannot be reported. The closest available measure is the value of cancelled "
     "and unavailable orders (ask: \"How much was lost to cancellations?\")."),
    ("profit and margin", r"\bprofit\w*|\bmargins?\b|\bebitda?\b|\bbreak.?even\b",
     "Profit and margins cannot be calculated: the dataset has revenue (payments and item prices) but no product costs or expenses."),
    ("costs", r"(?<!freight )(?<!shipping )(?<!delivery )\b(costs?|cogs|expenses?|overheads?|salar(y|ies)|taxe?s?)\b",
     "The dataset contains no cost, expense or tax data (only what customers paid, item prices and freight), so costs cannot be reported. "
     "Freight charged to customers is available (ask: \"How much freight was billed in 2018?\")."),
    ("product names", r"\bproduct names?\b|\bnames? of (the |each |these |those |our )?(top |best )?(products?|items?)\b|\b(products?|items?)'?s? (names|titles)\b|what (is|are) the products? called",
     "Product names are not in the dataset: products are identified only by a product id and a category. "
     "Ask e.g. \"Which products generated the highest sales?\" to get product ids with their categories."),
    ("customer names", r"\b(customer|client|buyer|seller)s?'? names?\b|\bnames? of (the |these |those |our )?(customers?|clients?|sellers?)\b|\b(e-?mail|phone numbers?|contact details|home address)",
     "Customer and seller names or contact details are not in the dataset (it is anonymised); customers are identified only by "
     "a 32-character customer_unique_id with their city and state."),
    ("stock levels", r"\b(stock levels?|inventory levels?|in stock|on hand|warehouse\w*|stock on hand)\b",
     "Stock and inventory levels are not in the dataset. What exists is sold units per category and the demand forecast "
     "(ask: \"What is the predicted demand for bed_bath_table?\")."),
    ("marketing data", r"\b(marketing|advertis\w*|ad spend|website (traffic|visits)|conversion rates?|page ?views|clicks|campaigns?)\b",
     "Marketing, advertising and website data are not in the dataset; only orders, items, payments, reviews, products, customers and sellers are."),
]


# --------------------------------------------------------------------------- helpers
def money(v) -> str:
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "n/a"
        return f"R$ {float(v):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


def count(v) -> str:
    try:
        return f"{int(round(float(v))):,}"
    except (TypeError, ValueError):
        return "n/a"


def pct(v, d: int = 2) -> str:
    try:
        return f"{float(v):.{d}f}%"
    except (TypeError, ValueError):
        return "n/a"


def to_int(word: str) -> int:
    return int(word) if word.isdigit() else WORDNUM[word]


def month_label(ym: str) -> str:
    return pd.Timestamp(ym + "-01").strftime("%B %Y")


def month_end(ym: str) -> str:
    return str(pd.Period(ym, freq="M").end_time.date())


def inline_params(sql: str, params) -> str:
    """Readable copy of a statement with its parameters filled in (display only; execution uses bound parameters)."""
    text = " ".join(sql.split())
    for p in params or ():
        lit = str(p) if isinstance(p, (int, float, np.integer, np.floating)) else "'" + str(p).replace("'", "''") + "'"
        text = text.replace("?", lit, 1)
    return text


@dataclass
class Period:
    start: str                    # 'YYYY-MM-DD' inclusive
    end: str                      # 'YYYY-MM-DD' inclusive
    label: str
    notes: list = field(default_factory=list)
    month_aligned: bool = True

    @property
    def start_month(self) -> str:
        return self.start[:7]

    @property
    def end_month(self) -> str:
        return self.end[:7]


def _month_period(start_ym: str, end_ym: str, label: str | None = None, notes=None) -> Period:
    if start_ym > end_ym:
        start_ym, end_ym = end_ym, start_ym
    if label is None:
        label = month_label(start_ym) if start_ym == end_ym else f"{month_label(start_ym)} to {month_label(end_ym)}"
    return Period(start_ym + "-01", month_end(end_ym), label, list(notes or []))


def _shift_month(ym: str, k: int) -> str:
    return str(pd.Period(ym, freq="M") + k)


def parse_period(low: str) -> Period | None:
    """Extract a time period from the question. Relative expressions are anchored to the end of the complete data."""
    # explicit dates
    days = sorted("-".join(d) for d in re.findall(r"\b(20\d\d)-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b", low))
    if days:
        return Period(days[0], days[-1], days[0] if days[0] == days[-1] else f"{days[0]} to {days[-1]}", month_aligned=False)
    note_now = ("There is no 'today' in this data: orders are complete only up to 2018-08-19, so the latest complete period was used.")
    if re.search(r"\b(today|right now|currently)\b", low):
        return Period(DATA_END, DATA_END, f"{DATA_END} (last complete day)", [note_now], False)
    if re.search(r"\byesterday\b", low):
        d = str((pd.Timestamp(DATA_END) - pd.Timedelta(days=1)).date())
        return Period(d, d, f"{d} (day before the last complete day)", [note_now], False)
    if re.search(r"\b(this|current) week\b|\blast week\b", low):
        s = str((pd.Timestamp(DATA_END) - pd.Timedelta(days=6)).date())
        return Period(s, DATA_END, f"the week {s} to {DATA_END} (last complete week)", [note_now], False)
    m = re.search(r"\b(last|past|previous)\s+" + NUM + r"\s+(days?|weeks?)\b", low)
    if m:
        n = to_int(m.group(2)) * (7 if m.group(3).startswith("week") else 1)
        s = str((pd.Timestamp(DATA_END) - pd.Timedelta(days=n - 1)).date())
        return Period(s, DATA_END, f"the {n} days {s} to {DATA_END}", [f"Counted back from {DATA_END}, the last complete day in the data."], False)
    m = re.search(r"\b(last|past|previous)\s+" + NUM + r"\s+months\b", low)
    if m:
        n = max(1, min(to_int(m.group(2)), 24))
        return _month_period(_shift_month(LAST_COMPLETE_MONTH, -(n - 1)), LAST_COMPLETE_MONTH,
                             notes=[f"'Last {n} months' = the {n} complete calendar months ending July 2018 (August 2018 is complete only to the 19th)."])
    if re.search(r"\b(last|previous|past) month\b", low):
        return _month_period(LAST_COMPLETE_MONTH, LAST_COMPLETE_MONTH,
                             notes=["'Last month' = July 2018, the last complete calendar month in the data (August 2018 is complete only to the 19th)."])
    if re.search(r"\b(this|current) month\b", low):
        return _month_period("2018-08", "2018-08", notes=["August 2018 is the latest month in the data but is complete only up to the 19th."])
    if re.search(r"\b(last|previous|past) quarter\b", low):
        return _month_period("2018-04", "2018-06", "Q2 2018 (April to June 2018)", ["'Last quarter' = Q2 2018, the last complete quarter in the data."])
    m = re.search(r"\bq([1-4])\b(?:\s*(?:of\s+)?(20\d\d))?|\b(first|second|third|fourth) quarter(?: of)?(?:\s+(20\d\d))?", low)
    if m:
        q = int(m.group(1)) if m.group(1) else ["first", "second", "third", "fourth"].index(m.group(3)) + 1
        year = m.group(2) or m.group(4) or ("2018" if q <= 2 else "2017")
        s = f"{year}-{3 * q - 2:02d}"
        return _month_period(s, _shift_month(s, 2), f"Q{q} {year}")
    if re.search(r"\b(last|previous|past) year\b", low):
        return _month_period("2017-01", "2017-12", "2017", ["'Last year' = 2017, the last complete calendar year in the data."])
    if re.search(r"\b(this|current) year\b|\byear to date\b|\bytd\b", low):
        return _month_period("2018-01", "2018-12", "2018", ["2018 is complete only up to 2018-08-19."])

    # month mentions ('YYYY-MM' and month names) and standalone years
    mentions = [(m.start(), int(m.group(1)), int(m.group(2))) for m in re.finditer(r"\b(20\d\d)-(0[1-9]|1[0-2])\b", low)]
    years_alone = [int(y) for y in re.findall(r"(?<![\d-])(20\d\d)(?![\d-])", low)]
    named = []
    for m in MONTH_RE.finditer(low):
        word = m.group(1)
        if word == "may" and not m.group(2) and not re.search(r"\b(in|of|for|during|from|to|and|until|since|between)\s+$", low[:m.start()]):
            continue                                               # "may" as a verb
        if word == "mar" and not m.group(2):
            continue
        named.append((m.start(), MONTH_NAMES[word[:3]], int(m.group(2)) if m.group(2) else None))
    if named:
        later_years = [y for _, _, y in named if y]
        for pos, mon, y in named:
            if y is None:
                if later_years:
                    y = later_years[-1]
                elif years_alone:
                    y = years_alone[-1]
                else:
                    y = 2018 if mon <= 8 else 2017                 # most recent occurrence inside the data
            mentions.append((pos, y, mon))
    if mentions:
        yms = sorted(f"{y}-{mo:02d}" for _, y, mo in mentions)
        return _month_period(yms[0], yms[-1])
    if years_alone:
        y0, y1 = min(years_alone), max(years_alone)
        return _month_period(f"{y0}-01", f"{y1}-12", str(y0) if y0 == y1 else f"{y0} to {y1}")
    return None


def period_notes(p: Period) -> list[str]:
    notes = list(p.notes)
    if p.end > DATA_END and p.start <= LAST_DAY and not any("complete only" in n for n in notes):
        notes.append("Orders after 2018-08-19 are incomplete in the source extract, so this period is under-counted.")
    if p.start < "2017-01-01" and p.end >= FIRST_DAY:
        notes.append("2016 contains very few orders (the extract starts in September 2016).")
    return notes


def parse_top_n(low: str, default: int = 10, cap: int = MAX_ROWS) -> tuple[int, bool]:
    m = (re.search(r"\b(?:top|first|bottom|last)\s+" + NUM + r"\b(?!\s*(days?|weeks?|months?|years?))", low)
         or re.search(r"\b" + NUM + r"\s+(?:best|top|highest|largest|biggest|most|lowest|worst|least|leading)\b", low))
    if m:
        n = to_int(m.group(1))
        if 1 <= n <= cap:
            return n, True
    return default, False


def parse_horizon_days(low: str) -> tuple[int, str]:
    """Forecast horizon in days (1-90) and a note when the request had to be capped."""
    m = (re.search(r"\b(?:next|coming|following|upcoming|in|within|over|for)\s+(?:the\s+)?(?:next\s+)?" + NUM + r"\s+(days?|weeks?|months?|years?)\b", low)
         or re.search(r"\b" + NUM + r"\s+(days?|weeks?|months?|years?)\s+(?:ahead|from now|forecast|prediction)\b", low))
    if m:
        n = to_int(m.group(1)) * {"d": 1, "w": 7, "m": 30, "y": 365}[m.group(2)[0]]
    elif re.search(r"\btomorrow\b|\bnext day\b", low):
        n = 1
    elif re.search(r"\b(next|coming|following|upcoming) (week)\b", low):
        n = 7
    elif re.search(r"\bfortnight\b", low):
        n = 14
    elif re.search(r"\b(next|coming|following|upcoming) quarter\b", low):
        n = 90
    elif re.search(r"\b(next|coming|following|upcoming) (year|12 months)\b", low):
        n = 365
    else:
        n = 30
    if n > 90:
        return 90, f"Forecasts are limited to 90 days, so the first 90 days (from {FORECAST_START}) are shown."
    return max(1, n), ""


# --------------------------------------------------------------------------- knowledge base
class KnowledgeBase:
    """TF-IDF retrieval over the sections ('## ' headings) of the markdown documents."""

    CONCEPT_DOCS = [
        (r"churn|risk categor|synthetic|retention|recency", "churn.md"),
        (r"ensemble|weight|five models|rank|xgboost|random forest|gradient boosting|decision tree|logistic|ridge|linear regression|recursive", "ensemble.md"),
        (r"net revenue|net sales|leakage|deduction|cancel|voucher|pending", "net_revenue.md"),
        (r"accura|mape|wape|rmse|r2|roc|auc|precision|recall|metric|evaluat", "agents_accuracy.md"),
        (r"limitation|caveat|missing|coverage|date range|time range|incomplete|data quality|what data|currency", "data_limitations.md"),
        (r"\btables?\b|\bviews?\b|glossary|schema|v_\w+|columns?|order_summary|payment_total|items_value", "glossary.md"),
    ]

    def __init__(self, folder: Path = KNOWLEDGE_DIR):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.chunks = []
        for path in sorted(folder.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            title = (re.search(r"^# (.+)$", text, re.M) or re.search(r"(.+)", text)).group(1).strip()
            for part in re.split(r"^## ", text, flags=re.M)[1:]:
                heading, _, body = part.partition("\n")
                self.chunks.append(dict(doc=path.name, title=title, heading=heading.strip(), body=body.strip()))
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        corpus = [f"{c['title']} {c['heading']} {c['heading']} {c['body']}" for c in self.chunks]
        self.matrix = self.vectorizer.fit_transform(corpus)

    def preferred_doc(self, low: str) -> str | None:
        for pattern, doc in self.CONCEPT_DOCS:
            if re.search(pattern, low):
                return doc
        return None

    def search(self, query: str, k: int = 3) -> list[tuple[float, float, dict]]:
        """Top chunks as (concept-boosted score, raw TF-IDF score, chunk)."""
        q = self.vectorizer.transform([query])
        scores = (self.matrix @ q.T).toarray().ravel()
        pref = self.preferred_doc(query.lower())
        boosted = [(float(s) + (0.25 if pref and c["doc"] == pref and s > 0 else 0.0), float(s), c)
                   for s, c in zip(scores, self.chunks)]
        boosted.sort(key=lambda t: t[0], reverse=True)
        return [(b, raw, c) for b, raw, c in boosted[:k] if raw > 0]


# --------------------------------------------------------------------------- per-answer context
class _Ctx:
    def __init__(self):
        self.text: list[str] = []
        self.tables: list[dict] = []
        self.sql: list[str] = []
        self.sources: list[str] = []
        self.kind = "unavailable"

    def say(self, *lines):
        self.text.extend(l for l in lines if l)

    def source(self, s: str):
        if s not in self.sources:
            self.sources.append(s)

    def table(self, title: str, df):
        if isinstance(df, pd.Series):
            df = df.to_frame().reset_index()
        if isinstance(df, pd.DataFrame) and not df.empty:
            self.tables.append(dict(title=title, data=df))

    def result(self) -> dict:
        return dict(answer="\n\n".join(self.text).strip(), kind=self.kind, tables=self.tables, sql=self.sql, sources=self.sources)


# --------------------------------------------------------------------------- chatbot
class Chatbot:
    """Hybrid-retrieval business chatbot. `ask` never raises."""

    def __init__(self, controller: Controller | None = None, use_llm: bool | None = None):
        self.c = controller or Controller()
        self._agent_cache: dict = {}
        self._meta: dict = {}
        self._kb: KnowledgeBase | None = None
        self._llm_client = None
        self.use_llm = self._llm_available() if use_llm is None else use_llm

    # ------------------------------------------------------------ public
    def ask(self, question) -> dict:
        ctx = _Ctx()
        try:
            text = "" if question is None else str(question)
            text = text.strip()
            truncated = len(text) > 1000
            text = text[:1000]
            self._route(text, ctx)
            if truncated and ctx.kind != "unavailable":
                ctx.say("_(Only the first 1,000 characters of the question were read.)_")
            return ctx.result()
        except RequestError as exc:
            return self._failure(ctx, f"The request could not be answered: {exc}")
        except FileNotFoundError as exc:
            return self._failure(ctx, f"A required file is missing (database or trained model): {exc}")
        except Exception as exc:  # noqa: BLE001 - the chatbot must never crash the UI
            return self._failure(ctx, f"Something went wrong while answering ({type(exc).__name__}: {exc}).")

    # ------------------------------------------------------------ infrastructure
    def _failure(self, ctx: _Ctx, message: str) -> dict:
        return dict(answer=message + "\n\n" + self._examples_text(), kind="unavailable", tables=[], sql=list(ctx.sql), sources=list(ctx.sources))

    @staticmethod
    def _examples_text() -> str:
        return "You can ask, for example:\n" + "\n".join(f"- {e}" for e in EXAMPLES)

    def _q(self, ctx: _Ctx, sql: str, params=(), source: str | None = None) -> pd.DataFrame:
        df = self.c.query(sql, tuple(params))
        ctx.sql.append(inline_params(sql, params))
        if source:
            ctx.source(f"database: {source}")
        return df

    def _agent(self, feature: str, **params) -> dict:
        key = (feature, tuple(sorted(params.items())))
        if key not in self._agent_cache:
            self._agent_cache[key] = self.c.run(feature, **params)
        return self._agent_cache[key]

    @property
    def kb(self) -> KnowledgeBase:
        if self._kb is None:
            self._kb = KnowledgeBase()
        return self._kb

    def _categories(self, ctx: _Ctx) -> list[str]:
        if "categories" not in self._meta:
            df = self._q(ctx, "SELECT category_name_english AS category FROM product_categories ORDER BY category_name_english",
                         source="product_categories")
            self._meta["categories"] = df.category.tolist()
        return self._meta["categories"]

    def _forecast_categories(self, ctx: _Ctx) -> list[str]:
        if "forecast_categories" not in self._meta:
            df = self._q(ctx, "SELECT DISTINCT category FROM demand_forecast ORDER BY category", source="demand_forecast")
            self._meta["forecast_categories"] = df.category.tolist()
        return self._meta["forecast_categories"]

    def find_category(self, norm: str, ctx: _Ctx) -> str | None:
        cats = self._categories(ctx)
        spaced = {c.replace("_", " "): c for c in cats if c != "unknown"}
        for name in sorted(spaced, key=len, reverse=True):
            if re.search(r"\b" + re.escape(name) + r"\b", norm):
                return spaced[name]
        for alias in sorted(CATEGORY_ALIASES, key=len, reverse=True):
            if re.search(r"\b" + re.escape(alias) + r"\b", norm) and CATEGORY_ALIASES[alias] in cats:
                return CATEGORY_ALIASES[alias]
        words = [w for w in re.findall(r"[a-z]+", norm)]
        names = list(spaced)
        for n in (4, 3, 2, 1):
            for i in range(len(words) - n + 1):
                gram_words = words[i:i + n]
                if all(w in FUZZY_STOP or len(w) < 4 for w in gram_words):
                    continue
                gram = " ".join(gram_words)
                if len(gram) < 5:
                    continue
                hit = difflib.get_close_matches(gram, names, n=1, cutoff=0.85)
                if hit:
                    return spaced[hit[0]]
        return None

    @staticmethod
    def find_state(text: str, norm: str) -> str | None:
        for code in STATES:
            if re.search(r"(?<![A-Za-z])" + code + r"(?![A-Za-z])", text) and code not in ("TO", "PA", "MA", "AM", "SE", "PE", "ES", "AL", "AC"):
                return code
        ascii_norm = norm.replace("ã", "a").replace("á", "a").replace("í", "i").replace("é", "e").replace("ô", "o")
        for code, name in sorted(STATES.items(), key=lambda kv: -len(kv[1])):
            if len(name) > 4 and re.search(r"\b" + name + r"\b", ascii_norm):
                return code
        return None

    # ------------------------------------------------------------ router
    def _route(self, text: str, ctx: _Ctx) -> None:
        low = text.lower()
        norm = re.sub(r"[_\-/]+", " ", low)
        if not re.search(r"[a-z0-9]", low):
            ctx.say("Please type a question about the business data.", self._examples_text())
            return
        if SQL_ATTACK.search(text):
            ctx.say("I can only read data and answer questions; I do not run database commands. Nothing was executed.", self._examples_text())
            return
        hexes = re.findall(r"\b[0-9a-f]{32}\b", low)
        if hexes:
            return self.h_entity(hexes[0], low, ctx)
        if re.fullmatch(r"\s*(hi|hello|hey|help|menu|start|what can you do\??|who are you\??|what can i ask\??)\s*[.!?]*\s*", low):
            ctx.kind = "explanation"
            ctx.say("I answer questions about the Olist e-commerce data (Sept 2016 - Oct 2018, complete 2017-01-09 to 2018-08-19): "
                    "historical facts from the database, forecasts from the four ML agents, and explanations of how things are defined.",
                    self._examples_text())
            return

        period = parse_period(low)
        pred = bool(re.search(PRED_W, low))
        net_ctx = bool(re.search(r"\bnet\b|cancel|leakage|\blost\b|\bloss(es)?\b|deduct", low))

        # 1. information that does not exist
        hits = [(name, msg) for name, pat, msg in UNAVAILABLE if re.search(pat, low)]
        if hits:
            if net_ctx and all(n in ("discounts", "returns", "refunds") for n, _ in hits):
                return self.h_net_revenue(low, period, ctx, pred, extra_missing=[n for n, _ in hits])
            ctx.kind = "unavailable"
            ctx.say(*[m for _, m in hits])
            ctx.say("See the data limitations: the dataset has orders, items, payments, reviews, products (id and category), "
                    "anonymised customers and sellers.")
            ctx.source("knowledge: data_limitations.md")
            return

        # 2. model accuracy and explanations
        if re.search(r"\baccura\w*|\bhow (good|well|reliable|precise|trustworthy)\b.*\b(model|forecast|prediction|agent|ensemble)s?\b|"
                     r"\breliab\w*|\b(mape|wape|rmse|r2|r squared|roc|auc|f1)\b|\b(precision|recall)\b|\bevaluat\w*|"
                     r"\b(model|forecast|prediction)s? (performance|errors?|quality)\b|\berror rates?\b|\b(validation|test) (scores?|results?|metrics?)\b", norm):
            return self.h_accuracy(low, ctx)
        wants_data = bool(re.search(r"\b(how (many|much)|total|rate|number of|count|list|show|top|which|who|predicted|expected|forecast for|value|amount|sum|by)\b", low)) \
            or period is not None
        definitional = bool(re.search(
            r"\b(explain|define|definition|meaning|methodology|limitations?|caveats?|glossary|data quality)\b|"
            r"\bhow (is|are|was|were|does|do|did) .{0,60}\b(calculat|comput|defin|work|measur|train|built|made|weight|combin|deriv|"
            r"decid|determin|creat|generat|scor|select|rank|forecast|predict|evaluat)\w*|"
            r"^(what|whats|what's)\s+(is|are|does|do)\s+(an?\s+|the\s+)?(churn|ensemble|weighted ensemble|"
            r"synthetic|risk categor\w*|churn probability|demand forecast|wape|mape|rmse|r2|roc|auc|precision|recall|v_\w+|"
            r"order_summary|payment_total|items_value|recency|returning customer|repeat customer)\w*\s*\??$|"
            r"^(what|whats|what's)\s+(is|are)\s+(an?\s+)?(net revenue|gross revenue|revenue|freight|vouchers?|leakage)\s*\??$|"
            r"\bwhat does .{1,40} mean\b|\bwhy (is|are|do|does|did|use|were)\b|\bwhich (tables|views)\b|\bwhat data\b", low))
        if definitional and not (wants_data and not re.search(r"\bhow (is|are|does|do)\b|\bexplain|\bdefin|\bwhy\b", low)):
            if self.h_knowledge(text, ctx):
                return

        # 3. whole-business analysis
        others = re.search(REVENUE_W + r"|demand|churn|customer|order|product|categor|state", low)
        if re.search(r"\b(business analysis|full analysis|complete analysis|summary of the business|business summary|overview|"
                     r"how (is|are) (the|our) business|how('s| is) business|executive summary|state of the business|health of the business|"
                     r"business report|key insights|recommendations?|what should (we|i) do|analy[sz]e (the|our) business|all agents)\b", low) \
                or (re.search(r"\boverall\b", low) and not others):
            return self.h_full_analysis(ctx)

        # 4. customers
        customers = bool(re.search(CUSTOMER_W, low))
        churn_w = bool(re.search(r"churn|at.risk|risk of (leaving|churn)|retention|retain|likely to (leave|stop|churn|return|come back)|"
                                 r"(lose|losing|lost) (customers|clients)|\bleaving\b|high.risk|probabilit", low))
        lapse_w = bool(re.search(r"\bnot (purchased|bought|ordered|shopped|been active|returned|come back|came back)|"
                                 r"\b(haven'?t|hasn'?t|have not|has not|didn'?t|did not)\b.{0,25}\b(purchas|buy|bought|order|shop|return)\w*|"
                                 r"\binactive|\blapsed|\bdormant|\bnot recently\b|\bno (recent )?(purchases?|orders?)|\bstopped (buying|purchasing|ordering)\b", low))
        if customers and lapse_w and not churn_w and not pred:
            return self.h_lapsed(low, ctx)
        if churn_w and (customers or "churn" in low):
            return self.h_churn(low, ctx)
        if re.search(r"\b(returning|repeat|recurring|came back|come back|loyal|more than (one|once)|multiple (orders|purchases)|"
                     r"(bought|ordered|purchased) (again|more than once)|second (order|purchase))\b", low) and (customers or "return" in low):
            return self.h_returning(low, ctx)
        if customers and re.search(r"\b(top|best|most valuable|biggest|highest|largest|spent the most|spend the most|spend most|big spenders|vip)\b", low) \
                and not re.search(r"how many", low):
            return self.h_top_customers(low, period, ctx)

        # 5. demand direction and forecasts
        cat = self.find_category(norm, ctx)
        decline = re.search(r"\b(declin\w*|decreas\w*|falling|fall\w*|drop\w*|shrink\w*|slow\w*|weak\w*|going down)\b", low)
        grow = re.search(r"\b(increas\w*|grow\w*|rising|rise|rises|gaining|trending up|going up|up.?trend)\b", low)
        demand_ctx = re.search(DEMAND_W + r"|categor|products?\b|sell\w*|items?\b", low)
        if (decline or grow) and re.search(REVENUE_W, low) and not re.search(r"categor|products?\b", low) and not customers:
            return self.h_revenue_forecast(low, period, ctx, target="net" if re.search(r"\bnet\b", low) else "gross")
        if (decline or grow) and demand_ctx and not customers and not re.search(REVENUE_W, low) or \
                ((decline or grow) and re.search(r"categor|products?\b", low) and not customers):
            return self.h_demand_direction(low, ctx, "decreasing" if decline else "increasing")
        if pred:
            if re.search(r"\bnet\b", low):
                return self.h_revenue_forecast(low, period, ctx, target="net")
            if cat or re.search(DEMAND_W + r"|categor|products?\b|\bsell\w*|\bsold\b|items?\b", low):
                return self.h_demand(low, cat, ctx, forecast=True)
            if re.search(REVENUE_W, low):
                return self.h_revenue_forecast(low, period, ctx, target="gross")
            if re.search(r"\b(orders?|customers?|reviews?|visitors?)\b", low):
                ctx.kind = "unavailable"
                ctx.say("No model forecasts order counts, customer counts or reviews. Forecasts exist for revenue, net revenue, "
                        "weekly demand of the 20 largest categories, and customer churn.", self._examples_text())
                return
            if re.search(r"\b(predict\w*|forecast\w*|projection\w*|outlook|future|expected)\b|\btomorrow\b|"
                         r"\bnext\s+(\d+\s+|\w+\s+)?(days?|weeks?|months?|quarter|year)\b", low):
                self.h_revenue_forecast(low, period, ctx, target="gross")
                ctx.say("No subject was named, so this is the revenue forecast. Forecasts also exist for net revenue, "
                        "weekly category demand and customer churn: ask, for example, 'demand forecast for next 4 weeks' or "
                        "'which customers are likely to churn?'.")
                return
        if cat and (re.search(DEMAND_W + r"|" + REVENUE_W + r"|\bsold\b|\bsell\w*|perform\w*|how (is|are|did|was)", low) or len(norm.split()) <= 4):
            forecast = bool(re.search(r"\bdemand\b", low)) and not re.search(PAST_W, low) and period is None
            return self.h_demand(low, cat, ctx, forecast=forecast, period=period)
        if re.search(r"\bdemand\b", low) and not re.search(r"\bproducts?\b", low):
            return self.h_demand(low, None, ctx, forecast=not re.search(PAST_W, low) and period is None, period=period)

        # 6. historical facts
        if net_ctx and re.search(r"\bnet\b|cancel|unavailable orders|leakage|\blost\b|\bloss(es)?\b|deduct", low):
            return self.h_net_revenue(low, period, ctx, pred=False)
        if re.search(r"\bproducts?\b|\bitems?\b|\bskus?\b|\b(best|top).?(selling|sellers? products?)\b", low) and not re.search(r"\bhow many items\b", low):
            return self.h_products(low, period, cat, ctx)
        if re.search(r"categor", low):
            return self.h_categories(low, period, ctx)
        if re.search(r"\bsellers?\b|\bvendors?\b|\bmerchants?\b", low):
            return self.h_sellers(low, period, ctx)
        state = self.find_state(text, norm)
        if re.search(r"\bstates?\b|\bregions?\b|\bby location\b|geograph\w*|\bwhere\b|\bcit(y|ies)\b", low) or state:
            return self.h_states(low, period, state, ctx)
        if re.search(r"\bstatus(es)?\b|\bdelivered orders\b", low):
            return self.h_breakdown(ctx, period, "status")
        if re.search(r"\breviews?\b|\bratings?\b|\bsatisf\w*|\bstars?\b", low):
            return self.h_breakdown(ctx, period, "reviews")
        if re.search(r"\bpayment (types?|methods?)\b|\bhow (do|did) (customers|people) pay\b|credit card|boleto|debit card|\bvouchers?\b|instal+ments?", low):
            return self.h_breakdown(ctx, period, "payments")
        if re.search(r"\bdelivery (time|days|speed|performance)\b|\blate (deliver\w*|orders?)\b|\bdelivered late\b|\bon.time\b|\bshipping time\b", low):
            return self.h_breakdown(ctx, period, "delivery")
        if re.search(REVENUE_W + r"|\bhow many (orders|customers|purchases)\b|\bnumber of (orders|customers)\b|\b(orders?|customers?) count\b|"
                     r"\baverage (order|basket|ticket)\b|\baov\b|\bfreight\b|\bshipping\b|\bperform\w*|\bhappen\w*|\bbusiness\b|\bhow many orders\b|\b(sell|sold)\b", low) \
                or (period is not None and re.search(r"\b(orders?|customers?|purchases?)\b", low)):
            return self.h_revenue_totals(low, period, ctx)

        # 7. knowledge fallback, LLM fallback, give up
        if self.h_knowledge(text, ctx, threshold=0.2, raw_threshold=0.2):
            return
        if self.use_llm and self.h_llm_sql(text, ctx):
            return
        ctx.kind = "unavailable"
        ctx.say("I could not match this to the business data. I answer questions about this e-commerce dataset only "
                "(revenue, net revenue, products and categories, states, customers and churn, demand and revenue forecasts, "
                "and how the models work).", self._examples_text())

    # ------------------------------------------------------------ handlers: knowledge
    def h_knowledge(self, text: str, ctx: _Ctx, threshold: float = 0.12, raw_threshold: float = 0.0) -> bool:
        """Answer from the knowledge documents. `threshold` applies to the concept-boosted score, `raw_threshold` to plain TF-IDF."""
        hits = self.kb.search(text, k=4)
        if not hits or hits[0][0] < threshold or hits[0][1] < raw_threshold:
            return False
        best_score, _, best = hits[0]
        broad = bool(re.search(r"limitation|caveat|tell me about|overview of", text.lower()))
        extra = hits[1:3] if broad else hits[1:2]
        chosen = [best] + [c for s, _, c in extra if s >= (0.8 if broad else 0.6) * best_score and c["heading"] != best["heading"]]
        ctx.kind = "explanation"
        for c in chosen:
            ctx.say(f"**{c['heading']}**\n\n{c['body']}")
            ctx.source(f"knowledge: {c['doc']}")
        low = text.lower()
        if best["doc"] == "ensemble.md" or re.search(r"\bweights?\b|ensemble", low):
            w = self._q(ctx, "SELECT DISTINCT agent, model, weight FROM model_evaluation WHERE weight IS NOT NULL ORDER BY agent, weight DESC",
                        source="model_evaluation")
            ctx.table("Ensemble weights per agent (from model_evaluation)", w)
            ctx.say("The table lists the weight each model received in each agent's ensemble.")
        if self.use_llm:
            self._llm_reword(text, ctx)
        return True

    def h_accuracy(self, low: str, ctx: _Ctx) -> None:
        agents = []
        if re.search(r"\bnet\b", low):
            agents.append("netrevenue")
        elif re.search(r"revenue|sales|turnover", low):
            agents.append("revenue")
        if re.search(r"demand|categor|units", low):
            agents.append("demand")
        if re.search(r"churn|customer", low):
            agents.append("churn")
        agents = agents or ["revenue", "netrevenue", "demand", "churn"]
        metrics = ["MAPE", "WAPE", "R2", "RMSE", "MAE", "ROC_AUC", "PR_AUC", "Precision", "Recall", "F1", "Accuracy"]
        cols = ", ".join(f"MAX(CASE WHEN metric = '{m}' THEN value END) AS {m}" for m in metrics)
        marks = ", ".join("?" for _ in agents)
        df = self._q(ctx, f"SELECT agent, split, model, weight, {cols} FROM model_evaluation WHERE agent IN ({marks}) "
                          "GROUP BY agent, split, model ORDER BY agent, split, model", agents, source="model_evaluation")
        df = df.dropna(axis=1, how="all")
        ctx.kind = "explanation"
        names = {"revenue": "Revenue", "netrevenue": "Net revenue", "demand": "Demand", "churn": "Churn"}
        lines = []
        for a in agents:
            d = df[(df.agent == a) & df.split.str.startswith("test")]
            ens = d[d.model == "ensemble"]
            if ens.empty:
                continue
            e = ens.iloc[0]
            if a == "churn":
                lines.append(f"- **{names[a]} model** (test snapshot, cutoff 2018-02-20): ROC-AUC {e.ROC_AUC:.3f}, precision {e.Precision:.3f}, "
                             f"recall {e.Recall:.3f}, F1 {e.F1:.3f}, accuracy {e.Accuracy:.3f}. It is trained on real plus synthetic repeat "
                             "orders, so these figures describe performance on the designed synthetic pattern.")
            else:
                naive = d[d.model.str.startswith("naive")]
                base = "; ".join(f"{r.model}: MAPE {r.MAPE:.1f}%, WAPE {r.WAPE:.1f}%" for r in naive.itertuples())
                unit = "weekly category units" if a == "demand" else "daily revenue"
                lines.append(f"- **{names[a]} model** ({unit}, test period): MAPE {e.MAPE:.1f}%, WAPE {e.WAPE:.1f}%, R2 {e.R2:.3f}"
                             + (f" - naive baselines: {base}." if base else "."))
        ctx.say("Tested accuracy of the weighted ensembles on held-out test data (from the `model_evaluation` table):", "\n".join(lines))
        ctx.say("MAPE/WAPE are percentage errors (lower is better); R2 and ROC-AUC closer to 1 are better. "
                "The table has validation and test metrics for every model.")
        ctx.table("Model evaluation (validation and test)", df)
        ctx.source("knowledge: agents_accuracy.md")

    # ------------------------------------------------------------ handlers: entities
    def h_entity(self, hid: str, low: str, ctx: _Ctx) -> None:
        cust = self._q(ctx, "SELECT customer_unique_id FROM customers WHERE customer_unique_id = ? "
                            "UNION SELECT customer_unique_id FROM orders WHERE customer_id = ? AND is_synthetic = 0", (hid, hid), source="customers")
        if not cust.empty:
            return self.h_customer(cust.customer_unique_id.iloc[0], hid, ctx)
        prod = self._q(ctx, "SELECT * FROM v_product_sales WHERE product_id = ?", (hid,), source="v_product_sales")
        exists = prod if not prod.empty else self._q(ctx, "SELECT p.product_id, c.category_name_english AS category FROM products p "
                                                           "JOIN product_categories c ON c.category_id = p.category_id WHERE p.product_id = ?",
                                                     (hid,), source="products")
        if not exists.empty:
            return self.h_product(hid, prod, exists, ctx)
        order = self._q(ctx, "SELECT order_id, customer_unique_id, order_status, order_purchase_timestamp, order_delivered_customer_date, "
                             "delivery_state, n_items, items_value, freight_value, payment_total, voucher_value, review_score "
                             "FROM v_sales WHERE order_id = ?", (hid,), source="v_sales")
        if order.empty and not self._q(ctx, "SELECT order_id FROM orders WHERE order_id = ? AND is_synthetic = 1", (hid,)).empty:
            ctx.kind = "unavailable"
            ctx.say(f"Order `{hid}` is a synthetic order generated for the churn model, not a real order, so it is not reported as a record.")
            return
        if not order.empty:
            r = order.iloc[0]
            ctx.kind = "historical"
            ctx.say(f"Order `{hid}` (real order): status **{r.order_status}**, placed {r.order_purchase_timestamp} by customer "
                    f"`{r.customer_unique_id}`, delivered to {r.delivery_state}; {count(r.n_items)} item(s), billed {money(r.payment_total)} "
                    f"(items {money(r.items_value)} + freight {money(r.freight_value)}).")
            ctx.table("Order record", order)
            return
        ctx.kind = "unavailable"
        ctx.say(f"`{hid}` is not a customer, product or order id in the database. Customer questions need the 32-character "
                "`customer_unique_id` (or the per-order `customer_id`).")

    def h_customer(self, cid: str, asked: str, ctx: _Ctx) -> None:
        hist = self._q(ctx, "SELECT * FROM v_customer_summary WHERE customer_unique_id = ?", (cid,), source="v_customer_summary")
        score = self._q(ctx, "SELECT * FROM churn_predictions WHERE customer_unique_id = ?", (cid,), source="churn_predictions")
        lines = []
        if asked != cid:
            lines.append(f"`{asked}` is a per-order customer_id; it belongs to customer `{cid}`.")
        if not score.empty:
            s = score.iloc[0]
            res = self._agent("churn", customer_unique_id=cid)
            ctx.source("agent: churn ensemble (models/churn.py)")
            lines.append(f"**Churn risk (model estimate, not a recorded fact)** for customer `{cid}`, scored at {s.snapshot_date}: "
                         f"churn probability **{s.churn_probability:.4f}** ({s.churn_probability * 100:.1f}%), risk category **{s.risk_category}**, "
                         f"predicted to churn: **{'yes' if int(s.predicted_churn) == 1 else 'no'}** (threshold 0.5). "
                         f"Churn = no order within 180 days after {s.snapshot_date}.")
            lines.append(f"The model saw {count(s.n_orders)} order(s), the last one {count(s.recency_days)} days before the snapshot. "
                         "Scores come from a model trained on real plus synthetic repeat orders, so its order history can include synthetic orders; "
                         "treat the score as a demonstration rather than evidence about the real customer.")
            ctx.table("Churn prediction (model output)", score)
            factors = res.get("factors")
            if isinstance(factors, pd.DataFrame):
                ctx.table("Main churn drivers of the model (permutation importance)", factors)
                lines.append(f"The model's strongest driver overall is '{factors.description.iloc[0]}'.")
        else:
            lines.append(f"Customer `{cid}` has no churn score: only customers who ordered before 2018-08-20 were scored.")
        if not hist.empty:
            h = hist.iloc[0]
            lines.append(f"**Real purchase history** (actual, real orders only): {count(h.n_orders)} order(s), first {h.first_purchase}, "
                         f"last {h.last_purchase}, total paid {money(h.total_paid)}, average review {h.avg_review_score if pd.notna(h.avg_review_score) else 'n/a'}, "
                         f"location {h.city} / {h.state}.")
            ctx.table("Customer history (real orders)", hist)
        ctx.kind = "mixed" if (not score.empty and not hist.empty) else ("prediction" if not score.empty else "historical")
        ctx.say(*lines)
        ctx.source("knowledge: churn.md")

    def h_product(self, pid: str, sales: pd.DataFrame, info: pd.DataFrame, ctx: _Ctx) -> None:
        ctx.kind = "historical"
        cat = info.category.iloc[0]
        if sales.empty:
            ctx.say(f"Product `{pid}` (category {cat}) has no sales in real, non-cancelled orders.")
            return
        r = sales.iloc[0]
        monthly = self._q(ctx, "SELECT substr(o.order_purchase_timestamp, 1, 7) AS month, COUNT(*) AS units, ROUND(SUM(i.price), 2) AS items_value "
                               "FROM order_items i JOIN orders o ON o.order_id = i.order_id WHERE i.product_id = ? AND o.is_synthetic = 0 "
                               "AND o.order_status NOT IN ('canceled', 'unavailable') GROUP BY month ORDER BY month", (pid,), source="order_items, orders")
        ctx.say(f"Product `{pid}` (category **{cat}**; product names are not in the dataset): {count(r.units)} units sold for "
                f"{money(r.items_value)} (average price {money(r.avg_price)}), first sale {r.first_sale}, last sale {r.last_sale}. "
                "Figures are actual sales from real orders, excluding cancelled and unavailable ones.",
                "No demand forecast exists for single products: they are too sparse to forecast, so demand is forecast only for the "
                "20 largest categories.")
        ctx.table("Product sales (all time)", sales)
        ctx.table("Monthly sales of the product", monthly)

    # ------------------------------------------------------------ handlers: customers
    def h_lapsed(self, low: str, ctx: _Ctx) -> None:
        m = re.search(NUM + r"\s+(days?|weeks?|months?|years?)", low)
        days = 180
        if m:
            days = to_int(m.group(1)) * {"d": 1, "w": 7, "m": 30, "y": 365}[m.group(2)[0]]
        days = max(1, min(days, 700))
        cutoff = str((pd.Timestamp(DATA_END) - pd.Timedelta(days=days - 1)).date())
        n, _ = parse_top_n(low, 20)
        summary = self._q(ctx, "SELECT SUM(last_purchase < ?) AS inactive_customers, COUNT(*) AS customers, "
                               "ROUND(100.0 * SUM(last_purchase < ?) / COUNT(*), 2) AS inactive_pct FROM v_customer_summary",
                          (cutoff, cutoff), source="v_customer_summary")
        rows = self._q(ctx, "SELECT customer_unique_id, city, state, n_orders, first_purchase, last_purchase, "
                            "CAST(julianday(?) - julianday(date(last_purchase)) AS INTEGER) AS days_since_last_purchase, total_paid "
                            "FROM v_customer_summary WHERE last_purchase < ? ORDER BY total_paid DESC LIMIT ?", (DATA_END, cutoff, n))
        s = summary.iloc[0]
        ctx.kind = "historical"
        ctx.say(f"**{count(s.inactive_customers)} of {count(s.customers)} customers ({pct(s.inactive_pct)})** have not placed a real order in the "
                f"{days} days up to {DATA_END} (the last complete day of data), i.e. their last purchase was before {cutoff}.",
                f"The table lists the {len(rows)} of them with the highest total spend (actual, real orders only). "
                "For a model estimate of who will not come back, ask \"Which customers have a high churn probability?\".")
        ctx.table(f"Customers with no purchase since {cutoff} (top {len(rows)} by total paid)", rows)

    def h_returning(self, low: str, ctx: _Ctx) -> None:
        s = self._q(ctx, "SELECT SUM(is_repeat_customer) AS returning_customers, COUNT(*) AS customers, "
                         "ROUND(100.0 * SUM(is_repeat_customer) / COUNT(*), 2) AS returning_pct FROM v_customer_summary",
                    source="v_customer_summary").iloc[0]
        dist = self._q(ctx, "SELECT n_orders, COUNT(*) AS customers FROM v_customer_summary GROUP BY n_orders ORDER BY n_orders")
        ctx.kind = "historical"
        ctx.say(f"**{count(s.returning_customers)} of {count(s.customers)} customers ({pct(s.returning_pct)})** are returning customers, "
                "i.e. placed more than one real order. This counts real orders only; the synthetic repeat orders made for the churn model are excluded.")
        ctx.table("Customers by number of real orders", dist)
        if re.search(r"\b(which|who|list|show|top)\b", low):
            n, _ = parse_top_n(low, 20)
            rows = self._q(ctx, "SELECT customer_unique_id, city, state, n_orders, first_purchase, last_purchase, total_paid "
                                "FROM v_customer_summary WHERE is_repeat_customer ORDER BY n_orders DESC, total_paid DESC LIMIT ?", (n,))
            ctx.table(f"Returning customers (top {len(rows)} by number of orders)", rows)
        ctx.source("knowledge: data_limitations.md")

    def h_top_customers(self, low: str, period: Period | None, ctx: _Ctx) -> None:
        n, _ = parse_top_n(low, 10)
        if period:
            rows = self._q(ctx, "SELECT customer_unique_id, delivery_state AS state, COUNT(*) AS orders, ROUND(SUM(payment_total), 2) AS total_paid "
                                "FROM v_sales WHERE purchase_date BETWEEN ? AND ? GROUP BY customer_unique_id ORDER BY total_paid DESC LIMIT ?",
                           (period.start, period.end, n), source="v_sales")
            label = period.label
        else:
            rows = self._q(ctx, "SELECT customer_unique_id, city, state, n_orders, first_purchase, last_purchase, total_paid "
                                "FROM v_customer_summary ORDER BY total_paid DESC LIMIT ?", (n,), source="v_customer_summary")
            label = "all time"
        if rows.empty:
            return self._no_data(ctx, period)
        r = rows.iloc[0]
        ctx.kind = "historical"
        ctx.say(f"Top {len(rows)} customers by amount paid ({label}, real orders). The largest is `{r.customer_unique_id}` with {money(r.total_paid)}. "
                "Customer names are not in the dataset, only anonymised ids.", *(period_notes(period) if period else []))
        ctx.table(f"Top customers by total paid ({label})", rows)

    def h_churn(self, low: str, ctx: _Ctx) -> None:
        n, explicit = parse_top_n(low, 20)
        res = self._agent("churn", top_n=n)
        ctx.source("agent: churn ensemble (models/churn.py)")
        ov = res["overview"]
        high = ov.loc[ov.risk_category == "High", "customers"].sum()
        ctx.kind = "prediction"
        ctx.say(f"**Churn model estimates** (scored at {res['snapshot_date']}; churn = no order within 180 days): "
                f"{count(high)} of {count(res['customers_scored'])} scored customers are **High risk** (churn probability >= 0.8), and "
                f"{count(res['predicted_to_churn'])} are predicted to churn (probability >= 0.5).")
        ctx.table("Churn risk overview (model output)", ov)
        if re.search(r"\b(which|who|list|show|top|customers with|customers that|customers who|high|most)\b", low) or explicit:
            rows = self._q(ctx, "SELECT customer_unique_id, churn_probability, risk_category, recency_days, n_orders, total_paid "
                                "FROM churn_predictions WHERE risk_category = 'High' ORDER BY churn_probability DESC, total_paid DESC LIMIT ?",
                           (n,), source="churn_predictions")
            if re.search(r"return|come back|stay|loyal|lowest|least likely", low):
                back = res["most_likely_to_return"]
                ctx.table(f"Customers least likely to churn (top {len(back)})", back)
                ctx.say(f"The 'least likely to churn' table shows the {len(back)} customers with the lowest churn probability, i.e. the model expects them "
                        f"to be most likely to order again (lowest score {back.churn_probability.iloc[0]:.4f}).")
            ctx.table(f"Customers with the highest churn probability (top {len(rows)})", rows)
            att = res["customers_needing_retention_attention"]
            ctx.table(f"Repeat customers predicted to churn, by past spend (agent's retention list, top {len(att)})", att)
            if not rows.empty:
                ctx.say(f"The 'highest churn probability' table lists {len(rows)} High-risk customers (top score "
                        f"{rows.churn_probability.iloc[0]:.4f}). The 'retention list' table is the agent's selection of repeat customers "
                        "predicted to churn, ordered by how much they have paid.")
        ctx.say("Note: these scores come from a model trained on real plus synthetic repeat orders (real data has too few returning "
                "customers to learn from), so they demonstrate the method rather than describe real customers. `recency_days`, "
                "`n_orders` and `total_paid` in the tables are the history the model saw, including synthetic orders.")
        ctx.source("knowledge: churn.md")

    # ------------------------------------------------------------ handlers: forecasts
    def h_revenue_forecast(self, low: str, period: Period | None, ctx: _Ctx, target: str = "gross") -> None:
        h, cap_note = parse_horizon_days(low)
        month_window = None
        if period and period.start > DATA_END:
            end_needed = (pd.Timestamp(period.end) - pd.Timestamp(FORECAST_START)).days + 1
            if end_needed > 90 and period.start <= FORECAST_LAST:
                ctx.say(f"Forecasts reach at most 90 days after the data ends ({FORECAST_START} to {FORECAST_LAST}), so only the part of "
                        f"{period.label} up to {FORECAST_LAST} is forecast below.")
                end_needed = 90
            if end_needed > 90:
                ctx.kind = "unavailable"
                ctx.say(f"There are no orders in {period.label}: the data ends on {LAST_DAY} (complete only to {DATA_END}). Forecasts reach at "
                        f"most 90 days after the data ends ({FORECAST_START} to {FORECAST_LAST}), so {period.label} cannot be forecast either.")
                return
            h, cap_note, month_window = end_needed, "", (period.start, period.end)
        if target == "net":
            res = self._agent("net_revenue", horizon_days=h)["forecast"]
            ctx.source("agent: net revenue ensemble (models/netrevenue.py)")
            what = "net revenue (billed minus cancelled and unavailable orders)"
        else:
            res = self._agent("revenue", horizon_days=h)
            ctx.source("agent: revenue ensemble (models/revenue.py)")
            what = "revenue (gross billed)"
        fc = res["forecast"]
        ctx.kind = "prediction"
        if month_window:
            part = fc[(fc.index >= month_window[0]) & (fc.index <= month_window[1])]
            ctx.say(f"**Forecast (model estimate)** of {what} for {period.label}: **{money(part.sum())}** "
                    f"({len(part)} forecast days, {str(part.index[0].date())} to {str(part.index[-1].date())}). "
                    "The source extract has only a handful of recorded orders after 2018-08-19, so this period is answered with the forecast.")
            table = part.rename("predicted_value").rename_axis("date").reset_index()
        else:
            label = "next month" if h == 30 else "next day" if h == 1 else f"next {h} days"
            ctx.say(f"**Forecast (model estimate)** of {what} for the {label}, {res['forecast_start']} to {res['forecast_end']} "
                    f"(the data ends on {DATA_END}): **{money(res['forecast_total'])}**.",
                    f"The model expects {res['growth_vs_previous_period_pct']:+.1f}% versus the previous {h} day{'s' if h > 1 else ''}, so the trend is **{res['trend']}**.")
            table = fc.rename("predicted_value").rename_axis("date").reset_index()
        ev = res["evaluation"]
        e = ev[(ev.split == "test") & (ev.model == "ensemble")]
        if not e.empty:
            ctx.say(f"Tested accuracy: one-day-ahead MAPE {e.MAPE.iloc[0]:.1f}% on the test period; over the 56-day test window the "
                    f"recursive forecast missed the total by {res['recursive_test']['total_error_pct']:+.1f}%. "
                    "History has one Black Friday only, so promotional spikes cannot be forecast.")
        ctx.say(cap_note)
        ctx.table(f"Daily {target} revenue forecast (model output)", table)
        ctx.source("knowledge: agents_accuracy.md")

    def h_demand(self, low: str, cat: str | None, ctx: _Ctx, forecast: bool, period: Period | None = None) -> None:
        top = self._forecast_categories(ctx)
        if forecast and (cat is None or cat in top):
            m = re.search(r"\b(?:next|coming|following)\s+" + NUM + r"\s+weeks?\b", low)
            if m:
                weeks = to_int(m.group(1))
            elif re.search(r"\bnext (two|2) months\b", low):
                weeks = 8
            elif re.search(r"\b(next|coming) (quarter|three months|3 months)\b", low):
                weeks = 12
            elif re.search(r"\bnext week\b", low):
                weeks = 1
            else:
                weeks = 4
            note = "Demand forecasts reach at most 12 weeks, so 12 weeks are shown." if weeks > 12 else ""
            weeks = max(1, min(weeks, 12))
            res = self._agent("demand", weeks=weeks, category=cat)
            ctx.source("agent: demand ensemble (models/demand.py)")
            s = res["summary"]
            ctx.kind = "prediction"
            period_txt = f"{res['forecast_start']} to {res['forecast_end']} ({weeks} week{'s' if weeks > 1 else ''})"
            if cat:
                r = s.iloc[0]
                ctx.say(f"**Demand forecast (model estimate)** for **{cat}**, {period_txt}: **{count(r.forecast_total_units)} units** in total, "
                        f"about {r.forecast_avg_weekly_units:.1f} units per week, {r.change_pct:+.1f}% versus the average of the last 4 weeks "
                        f"(trend: {r.trend}, demand level: {r.demand_level}).",
                        f"Recommendation: {r.recommendation}")
                ctx.table(f"Demand forecast summary: {cat}", s)
                ctx.table(f"Weekly forecast units: {cat}", res["weekly_forecast"].reset_index())
            else:
                ctx.say(f"**Demand forecast (model estimate)** for the 20 largest categories, {period_txt}.",
                        f"High demand: {', '.join(res['high_demand']) or 'none'}. Increasing (>10% above the last 4 weeks): "
                        f"{', '.join(res['increasing']) or 'none'}. Decreasing (>10% below): {', '.join(res['decreasing']) or 'none'}.")
                if re.search(r"\bproducts?\b", low):
                    ctx.say("Single products are too sparse to forecast, so demand is forecast per category.")
                ctx.table("Demand forecast by category (model output)", s)
                ctx.table("Weekly forecast units by category", res["weekly_forecast"].reset_index())
            ctx.say("Units = items sold in real orders, excluding cancelled and unavailable orders.", note)
            ctx.source("knowledge: agents_accuracy.md")
            return
        # historical demand (requested, or no forecast exists for this category)
        if cat is None:
            return self.h_categories(low, period, ctx, by_units=True)
        where, params = "", [cat]
        if period:
            where, params = " AND date(o.order_purchase_timestamp) BETWEEN ? AND ?", [cat, period.start, period.end]
        monthly = self._q(ctx, "SELECT substr(o.order_purchase_timestamp, 1, 7) AS month, COUNT(*) AS units, ROUND(SUM(i.price), 2) AS items_value "
                               "FROM order_items i JOIN orders o ON o.order_id = i.order_id JOIN products p ON p.product_id = i.product_id "
                               "JOIN product_categories c ON c.category_id = p.category_id WHERE c.category_name_english = ? AND o.is_synthetic = 0 "
                               "AND o.order_status NOT IN ('canceled', 'unavailable')" + where + " GROUP BY month ORDER BY month",
                          params, source="order_items, orders, products, product_categories")
        tot = self._q(ctx, "SELECT COUNT(*) AS units, ROUND(SUM(i.price), 2) AS items_value, COUNT(DISTINCT i.order_id) AS orders "
                           "FROM order_items i JOIN orders o ON o.order_id = i.order_id JOIN products p ON p.product_id = i.product_id "
                           "JOIN product_categories c ON c.category_id = p.category_id WHERE c.category_name_english = ? AND o.is_synthetic = 0 "
                           "AND o.order_status NOT IN ('canceled', 'unavailable')" + where, params).iloc[0]
        label = period.label if period else "all time"
        ctx.kind = "historical"
        if monthly.empty:
            return self._no_data(ctx, period, f"No sales of {cat} in {label}.")
        lines = [f"**Actual sales of {cat}** ({label}, real orders excluding cancelled and unavailable): {count(tot.units)} units in "
                 f"{count(tot.orders)} orders, item value {money(tot['items_value'])} (item prices, excluding freight)."]
        last = monthly[monthly.month == LAST_COMPLETE_MONTH]
        if not period and not last.empty:
            lines.append(f"In July 2018 (last complete month): {count(last.units.iloc[0])} units, {money(last['items_value'].iloc[0])}.")
        if forecast or re.search(PRED_W + r"|\bdemand\b", low):
            if cat not in top:
                lines.append(f"No demand forecast exists for {cat}: forecasts cover only the 20 largest categories "
                             f"({', '.join(top)}), so the historical sales are shown instead.")
        ctx.say(*lines, *(period_notes(period) if period else []))
        ctx.table(f"Monthly sales of {cat} ({label})", monthly)

    def h_demand_direction(self, low: str, ctx: _Ctx, direction: str) -> None:
        res = self._agent("demand", weeks=4)
        ctx.source("agent: demand ensemble (models/demand.py)")
        s = res["summary"]
        sel = s[s.trend == direction]
        ctx.kind = "prediction"
        word = "declining" if direction == "decreasing" else "growing"
        if sel.empty:
            ctx.say(f"**Demand forecast (model estimate)**, {res['forecast_start']} to {res['forecast_end']}: no category is forecast to be "
                    f"{word} by more than 10% versus its last 4 weeks.")
        else:
            items = ", ".join(f"{r.category} ({r.change_pct:+.1f}%)" for r in sel.itertuples())
            ctx.say(f"**Demand forecast (model estimate)** for {res['forecast_start']} to {res['forecast_end']}: {len(sel)} of the 20 largest "
                    f"categories are forecast to be **{word}** (forecast weekly average more than 10% {'below' if direction == 'decreasing' else 'above'} "
                    f"the last 4 weeks): {items}.")
            ctx.table(f"Categories with {direction} forecast demand", sel)
        if re.search(r"\bproducts?\b", low):
            ctx.say("Single products are too sparse to forecast, so the trend is given per category.")
        ctx.table("Demand forecast, all forecast categories", s)
        ctx.source("knowledge: agents_accuracy.md")

    # ------------------------------------------------------------ handlers: historical
    def _no_data(self, ctx: _Ctx, period: Period | None, msg: str | None = None) -> None:
        ctx.kind = "unavailable"
        ctx.say(msg or f"There are no orders in {period.label if period else 'that period'}. The data runs from {FIRST_DAY} to {LAST_DAY} "
                       f"(complete {DATA_START} to {DATA_END}).")

    @staticmethod
    def _out_of_range(period: Period | None) -> bool:
        return period is not None and (period.end < FIRST_DAY or period.start > LAST_DAY)

    def h_revenue_totals(self, low: str, period: Period | None, ctx: _Ctx) -> None:
        if period and period.start > DATA_END and re.search(REVENUE_W, low):
            return self.h_revenue_forecast(low, period, ctx)
        if self._out_of_range(period):
            return self._no_data(ctx, period)
        where, params = ("WHERE purchase_date BETWEEN ? AND ?", (period.start, period.end)) if period else ("", ())
        t = self._q(ctx, "SELECT COUNT(*) AS orders, COUNT(DISTINCT customer_unique_id) AS customers, ROUND(SUM(payment_total), 2) AS revenue, "
                         "ROUND(SUM(items_value), 2) AS items_value, ROUND(SUM(freight_value), 2) AS freight_value, "
                         "ROUND(AVG(payment_total), 2) AS avg_order_value, MIN(purchase_date) AS first_day, MAX(purchase_date) AS last_day "
                         f"FROM v_sales {where}", params, source="v_sales")
        r = t.iloc[0]
        if not r.orders:
            return self._no_data(ctx, period)
        label = period.label if period else f"all time ({r.first_day} to {r.last_day})"
        ctx.kind = "historical"
        lead = {
            "orders": f"There were **{count(r.orders)} orders** in {label}",
            "customers": f"**{count(r.customers)} distinct customers** ordered in {label}",
            "aov": f"The average order value in {label} was **{money(r.avg_order_value)}**",
            "freight": f"Freight billed in {label} was **{money(r.freight_value)}**",
        }
        if re.search(r"\b(how many|number of|count)\b.{0,15}\b(orders|purchases)\b|\borders count\b", low):
            first = lead["orders"] + f", billing {money(r.revenue)} in total."
        elif re.search(r"\b(how many|number of|count)\b.{0,15}\bcustomers\b", low):
            first = lead["customers"] + f" (placing {count(r.orders)} orders)."
        elif re.search(r"\baverage (order|basket|ticket)|\baov\b", low):
            first = lead["aov"] + f" ({count(r.orders)} orders, revenue {money(r.revenue)})."
        elif re.search(r"\bfreight\b|\bshipping\b", low):
            first = lead["freight"] + f", out of total revenue {money(r.revenue)}."
        else:
            first = (f"Total revenue for {label} was **{money(r.revenue)}** from {count(r.orders)} orders by {count(r.customers)} customers "
                     f"(average order value {money(r.avg_order_value)}).")
        ctx.say(first,
                f"Revenue = everything billed to customers (sum of payments) for real orders, including freight ({money(r.freight_value)}) "
                f"and regardless of order status; item prices alone total {money(r['items_value'])}. Net revenue (excluding cancelled and "
                "unavailable orders) is available on request.", *(period_notes(period) if period else []))
        ctx.table(f"Totals ({label})", t)
        mwhere, mparams = ("WHERE purchase_month BETWEEN ? AND ?", (period.start_month, period.end_month)) if period else ("", ())
        monthly = self._q(ctx, f"SELECT * FROM v_monthly_revenue {mwhere} ORDER BY purchase_month", mparams, source="v_monthly_revenue")
        if len(monthly) > 1 or not period:
            ctx.table("Monthly revenue", monthly)
        ctx.source("knowledge: data_limitations.md")

    def h_net_revenue(self, low: str, period: Period | None, ctx: _Ctx, pred: bool, extra_missing=None) -> None:
        if pred:
            return self.h_revenue_forecast(low, period, ctx, target="net")
        if self._out_of_range(period):
            return self._no_data(ctx, period)
        if period is None or period.month_aligned:
            where, params = ("WHERE purchase_month BETWEEN ? AND ?", (period.start_month, period.end_month)) if period else ("", ())
            monthly = self._q(ctx, f"SELECT * FROM v_net_revenue_monthly {where} ORDER BY purchase_month", params, source="v_net_revenue_monthly")
            t = self._q(ctx, "SELECT ROUND(SUM(gross_billed), 2) AS gross_billed, ROUND(SUM(cancelled_value), 2) AS cancelled_value, "
                             "ROUND(SUM(unavailable_value), 2) AS unavailable_value, ROUND(SUM(cancelled_value) + SUM(unavailable_value), 2) AS total_lost, "
                             "ROUND(SUM(net_revenue), 2) AS net_revenue, ROUND(SUM(delivered_value), 2) AS delivered_value, "
                             "ROUND(SUM(pending_value), 2) AS pending_value, ROUND(SUM(voucher_value), 2) AS voucher_value, "
                             "ROUND(100.0 * (SUM(cancelled_value) + SUM(unavailable_value)) / SUM(gross_billed), 2) AS lost_pct_of_billed "
                             f"FROM v_net_revenue_monthly {where}", params)
        else:
            monthly = pd.DataFrame()
            t = self._q(ctx, "SELECT ROUND(SUM(payment_total), 2) AS gross_billed, "
                             "ROUND(SUM(CASE WHEN order_status = 'canceled' THEN payment_total ELSE 0 END), 2) AS cancelled_value, "
                             "ROUND(SUM(CASE WHEN order_status = 'unavailable' THEN payment_total ELSE 0 END), 2) AS unavailable_value, "
                             "ROUND(SUM(CASE WHEN order_status IN ('canceled', 'unavailable') THEN payment_total ELSE 0 END), 2) AS total_lost, "
                             "ROUND(SUM(CASE WHEN order_status NOT IN ('canceled', 'unavailable') THEN payment_total ELSE 0 END), 2) AS net_revenue, "
                             "ROUND(SUM(CASE WHEN order_status = 'delivered' THEN payment_total ELSE 0 END), 2) AS delivered_value, "
                             "ROUND(SUM(CASE WHEN order_status IN ('created', 'approved', 'invoiced', 'processing', 'shipped') THEN payment_total ELSE 0 END), 2) AS pending_value, "
                             "ROUND(SUM(voucher_value), 2) AS voucher_value, "
                             "ROUND(100.0 * SUM(CASE WHEN order_status IN ('canceled', 'unavailable') THEN payment_total ELSE 0 END) / SUM(payment_total), 2) AS lost_pct_of_billed "
                             "FROM v_sales WHERE purchase_date BETWEEN ? AND ?", (period.start, period.end), source="v_sales")
        r = t.iloc[0]
        if pd.isna(r.gross_billed):
            return self._no_data(ctx, period)
        n = self._q(ctx, "SELECT SUM(order_status = 'canceled') AS cancelled_orders, SUM(order_status = 'unavailable') AS unavailable_orders, "
                         "COUNT(*) AS orders FROM v_sales" + (" WHERE purchase_date BETWEEN ? AND ?" if period else ""),
                    (period.start, period.end) if period else (), source="v_sales").iloc[0]
        label = period.label if period else "all time"
        ctx.kind = "historical"
        missing = ["discounts", "returns", "refunds"]
        ctx.say(f"**Net revenue for {label}: {money(r.net_revenue)}** = gross billed {money(r.gross_billed)} - cancelled orders "
                f"{money(r.cancelled_value)} - unavailable (out-of-stock) orders {money(r.unavailable_value)}.",
                f"Lost to cancellations: **{money(r.cancelled_value)}** ({count(n.cancelled_orders)} orders); lost to unavailable orders: "
                f"{money(r.unavailable_value)} ({count(n.unavailable_orders)} orders); together {money(r.total_lost)} "
                f"({pct(r.lost_pct_of_billed)} of the amount billed, {count(n.orders)} orders in total).",
                f"Of the net revenue, {money(r.delivered_value)} was delivered and {money(r.pending_value)} was still in progress at the end of the data. "
                f"Voucher payments ({money(r.voucher_value)}) are a payment method and are not deducted.",
                f"Not available in the dataset: {', '.join(missing)}"
                + (f" (you asked about {', '.join(extra_missing)})" if extra_missing else "")
                + ", so the true net revenue is lower by an unknown amount.", *(period_notes(period) if period else []))
        ctx.table(f"Net revenue components ({label})", t)
        ctx.table("Net revenue by month", monthly)
        ctx.source("knowledge: net_revenue.md")

    def _item_filter(self, period: Period | None, cat: str | None) -> tuple[str, list]:
        where, params = "WHERE o.is_synthetic = 0 AND o.order_status NOT IN ('canceled', 'unavailable')", []
        if period:
            where += " AND date(o.order_purchase_timestamp) BETWEEN ? AND ?"
            params += [period.start, period.end]
        if cat:
            where += " AND c.category_name_english = ?"
            params.append(cat)
        return where, params

    @staticmethod
    def _rank(low: str, value_col: str = "items_value") -> tuple[str, str, str]:
        by_units = bool(re.search(r"\b(units?|quantity|volume|most sold|sold the most|best.?sell\w*|top.?sell\w*|popular|number of (items|units))\b", low))
        asc = bool(re.search(r"\b(least|lowest|worst|bottom|fewest|smallest)\b", low))
        col = "units" if by_units else value_col
        return col, "ASC" if asc else "DESC", ("units sold" if by_units else "sales value")

    def h_products(self, low: str, period: Period | None, cat: str | None, ctx: _Ctx) -> None:
        if self._out_of_range(period):
            return self._no_data(ctx, period)
        n, _ = parse_top_n(low, 10)
        col, order, metric = self._rank(low)
        if not period and not cat:
            rows = self._q(ctx, f"SELECT * FROM v_product_sales ORDER BY {col} {order} LIMIT ?", (n,), source="v_product_sales")
        else:
            where, params = self._item_filter(period, cat)
            rows = self._q(ctx, "SELECT p.product_id, c.category_name_english AS category, COUNT(*) AS units, ROUND(SUM(i.price), 2) AS items_value, "
                                "ROUND(AVG(i.price), 2) AS avg_price FROM order_items i JOIN orders o ON o.order_id = i.order_id "
                                "JOIN products p ON p.product_id = i.product_id JOIN product_categories c ON c.category_id = p.category_id "
                                f"{where} GROUP BY p.product_id ORDER BY {col} {order} LIMIT ?", params + [n],
                           source="order_items, orders, products, product_categories")
        if rows.empty:
            return self._no_data(ctx, period)
        label = ", ".join(x for x in [period.label if period else "all time", f"category {cat}" if cat else ""] if x)
        top3 = "; ".join(f"`{r.product_id}` ({r.category}): {money(r.items_value)}, {count(r.units)} units" for r in rows.head(3).itertuples())
        ctx.kind = "historical"
        ctx.say(f"{'Bottom' if order == 'ASC' else 'Top'} {len(rows)} products by {metric} ({label}). Leading: {top3}.",
                "Sales value = sum of item prices (excluding freight) in real orders that were not cancelled or unavailable. "
                "Product names are not in the dataset; products are identified by id and category.", *(period_notes(period) if period else []))
        ctx.table(f"Products by {metric} ({label})", rows)

    def h_categories(self, low: str, period: Period | None, ctx: _Ctx, by_units: bool = False) -> None:
        if self._out_of_range(period):
            return self._no_data(ctx, period)
        n, explicit = parse_top_n(low, 74)
        col, order, metric = self._rank(low + (" units" if by_units else ""))
        where, params = self._item_filter(period, None)
        rows = self._q(ctx, "SELECT c.category_name_english AS category, COUNT(*) AS units, ROUND(SUM(i.price), 2) AS items_value, "
                            "COUNT(DISTINCT i.order_id) AS orders FROM order_items i JOIN orders o ON o.order_id = i.order_id "
                            "JOIN products p ON p.product_id = i.product_id JOIN product_categories c ON c.category_id = p.category_id "
                            f"{where} GROUP BY c.category_name_english ORDER BY {col} {order} LIMIT ?", params + [n],
                       source="order_items, orders, products, product_categories")
        if rows.empty:
            return self._no_data(ctx, period)
        label = period.label if period else "all time"
        shown = rows.head(n if explicit else 5)
        lst = "; ".join(f"{i + 1}. {r.category}: {money(r.items_value)} ({count(r.units)} units)" for i, r in enumerate(shown.itertuples()))
        ctx.kind = "historical"
        ctx.say(f"{'Bottom' if order == 'ASC' else 'Top'} {len(shown)} categories by {metric} ({label}): {lst}.",
                "Category sales = sum of item prices (excluding freight) in real orders that were not cancelled or unavailable.",
                *(period_notes(period) if period else []))
        ctx.table(f"Categories by {metric} ({label})", rows)

    def h_states(self, low: str, period: Period | None, state: str | None, ctx: _Ctx) -> None:
        if self._out_of_range(period):
            return self._no_data(ctx, period)
        n, explicit = parse_top_n(low, 27)
        where, params = ["1 = 1"], []
        if period:
            where.append("purchase_date BETWEEN ? AND ?")
            params += [period.start, period.end]
        if state:
            where.append("delivery_state = ?")
            params.append(state)
        asc = "ASC" if re.search(r"\b(least|lowest|worst|bottom|smallest)\b", low) else "DESC"
        rows = self._q(ctx, "SELECT delivery_state AS state, COUNT(*) AS orders, COUNT(DISTINCT customer_unique_id) AS customers, "
                            "ROUND(SUM(payment_total), 2) AS revenue, ROUND(AVG(payment_total), 2) AS avg_order_value "
                            f"FROM v_sales WHERE {' AND '.join(where)} GROUP BY delivery_state ORDER BY revenue {asc} LIMIT ?",
                       params + [n], source="v_sales")
        if rows.empty:
            return self._no_data(ctx, period)
        label = period.label if period else "all time"
        ctx.kind = "historical"
        if state:
            r = rows.iloc[0]
            ctx.say(f"Revenue from orders delivered to **{state}** ({STATES[state].title()}), {label}: **{money(r.revenue)}** from "
                    f"{count(r.orders)} orders by {count(r.customers)} customers (average order value {money(r.avg_order_value)}).")
        else:
            lst = "; ".join(f"{r.state}: {money(r.revenue)}" for r in rows.head(5).itertuples())
            ctx.say(f"Revenue by delivery state ({label}), {'lowest' if asc == 'ASC' else 'highest'} first: {lst}. The table has "
                    f"{'the requested' if explicit else 'all'} {len(rows)} states.")
        if re.search(r"\bcit(y|ies)\b", low):
            ctx.say("Revenue is reported by state; city-level revenue is not broken out by this chatbot.")
        ctx.say("Revenue = sum of payments for real orders, by the state the order was delivered to.", *(period_notes(period) if period else []))
        ctx.table(f"Revenue by state ({label})", rows)

    def h_sellers(self, low: str, period: Period | None, ctx: _Ctx) -> None:
        if self._out_of_range(period):
            return self._no_data(ctx, period)
        n, _ = parse_top_n(low, 10)
        where, params = self._item_filter(period, None)
        rows = self._q(ctx, "SELECT i.seller_id, s.state, COUNT(*) AS units, ROUND(SUM(i.price), 2) AS items_value, COUNT(DISTINCT i.order_id) AS orders "
                            "FROM order_items i JOIN orders o ON o.order_id = i.order_id JOIN sellers s ON s.seller_id = i.seller_id "
                            "JOIN products p ON p.product_id = i.product_id JOIN product_categories c ON c.category_id = p.category_id "
                            f"{where} GROUP BY i.seller_id ORDER BY items_value DESC LIMIT ?", params + [n], source="order_items, orders, sellers")
        total = self._q(ctx, "SELECT COUNT(*) AS sellers FROM sellers", source="sellers").iloc[0]
        label = period.label if period else "all time"
        ctx.kind = "historical"
        ctx.say(f"There are {count(total.sellers)} sellers in the database. Top {len(rows)} sellers by sales value ({label}); the largest, "
                f"`{rows.seller_id.iloc[0]}`, sold {money(rows['items_value'].iloc[0])}. Seller names are not in the dataset."
                if not rows.empty else f"There are {count(total.sellers)} sellers in the database; none sold in {label}.",
                *(period_notes(period) if period else []))
        ctx.table(f"Top sellers by sales value ({label})", rows)

    def h_breakdown(self, ctx: _Ctx, period: Period | None, what: str) -> None:
        if self._out_of_range(period):
            return self._no_data(ctx, period)
        label = period.label if period else "all time"
        pw, pp = ("purchase_date BETWEEN ? AND ?", [period.start, period.end]) if period else ("1 = 1", [])
        ctx.kind = "historical"
        if what == "status":
            rows = self._q(ctx, f"SELECT order_status, COUNT(*) AS orders, ROUND(SUM(payment_total), 2) AS billed FROM v_sales WHERE {pw} "
                                "GROUP BY order_status ORDER BY orders DESC", pp, source="v_sales")
            lst = "; ".join(f"{r.order_status}: {count(r.orders)}" for r in rows.itertuples())
            ctx.say(f"Orders by status ({label}, real orders): {lst}.")
            ctx.table(f"Orders by status ({label})", rows)
        elif what == "reviews":
            s = self._q(ctx, f"SELECT ROUND(AVG(review_score), 2) AS avg_review_score, COUNT(review_score) AS reviewed_orders, COUNT(*) AS orders "
                             f"FROM v_sales WHERE {pw}", pp, source="v_sales")
            dist = self._q(ctx, f"SELECT review_score, COUNT(*) AS orders FROM v_sales WHERE {pw} AND review_score IS NOT NULL "
                                "GROUP BY review_score ORDER BY review_score", pp)
            r = s.iloc[0]
            ctx.say(f"Average review score ({label}): **{r.avg_review_score}** out of 5, from {count(r.reviewed_orders)} reviewed orders "
                    f"(of {count(r.orders)} real orders).")
            ctx.table(f"Review summary ({label})", s)
            ctx.table(f"Orders by review score ({label})", dist)
        elif what == "payments":
            ow = "o.is_synthetic = 0" + (" AND date(o.order_purchase_timestamp) BETWEEN ? AND ?" if period else "")
            rows = self._q(ctx, "SELECT p.payment_type, COUNT(DISTINCT p.order_id) AS orders, COUNT(*) AS payments, ROUND(SUM(p.payment_value), 2) AS value, "
                                "ROUND(AVG(p.payment_installments), 2) AS avg_installments FROM payments p JOIN orders o ON o.order_id = p.order_id "
                                f"WHERE {ow} GROUP BY p.payment_type ORDER BY value DESC", pp, source="payments, orders")
            lst = "; ".join(f"{r.payment_type}: {money(r.value)} ({count(r.orders)} orders)" for r in rows.itertuples())
            ctx.say(f"Payments by type ({label}, real orders): {lst}. Vouchers are a payment method; the dataset has no discount data.")
            ctx.table(f"Payments by type ({label})", rows)
        elif what == "delivery":
            s = self._q(ctx, "SELECT COUNT(*) AS delivered_orders, ROUND(AVG(julianday(order_delivered_customer_date) - julianday(order_purchase_timestamp)), 1) "
                             "AS avg_delivery_days, ROUND(100.0 * AVG(order_delivered_customer_date > order_estimated_delivery_date), 2) AS late_pct "
                             f"FROM v_sales WHERE order_status = 'delivered' AND order_delivered_customer_date IS NOT NULL AND {pw}", pp, source="v_sales")
            r = s.iloc[0]
            ctx.say(f"Delivered orders ({label}): {count(r.delivered_orders)}; average time from purchase to delivery **{r.avg_delivery_days} days**; "
                    f"{pct(r.late_pct)} were delivered after the estimated delivery date.")
            ctx.table(f"Delivery performance ({label})", s)
        ctx.say(*(period_notes(period) if period else []))

    def h_full_analysis(self, ctx: _Ctx) -> None:
        k = self._q(ctx, "SELECT COUNT(*) AS total_orders, COUNT(DISTINCT customer_unique_id) AS total_customers, "
                         "ROUND(SUM(payment_total), 2) AS total_revenue, "
                         "ROUND(SUM(CASE WHEN order_status NOT IN ('canceled', 'unavailable') THEN payment_total ELSE 0 END), 2) AS net_revenue, "
                         "ROUND(AVG(payment_total), 2) AS avg_order_value, ROUND(AVG(review_score), 2) AS avg_review_score, "
                         "MIN(purchase_date) AS first_order_date, MAX(purchase_date) AS last_order_date FROM v_sales", source="v_sales")
        ret = self._q(ctx, "SELECT SUM(is_repeat_customer) AS returning_customers FROM v_customer_summary", source="v_customer_summary")
        july = self._q(ctx, "SELECT purchase_month, payment_total FROM v_monthly_revenue WHERE purchase_month = ?", (LAST_COMPLETE_MONTH,),
                       source="v_monthly_revenue")
        rev, net = self._agent("revenue", horizon_days=30), self._agent("net_revenue")
        dem, chn = self._agent("demand", weeks=4), self._agent("churn", top_n=20)
        for s in ("revenue", "net revenue", "demand", "churn"):
            ctx.source(f"agent: {s} ensemble")
        recs = Controller.recommendations(dict(revenue=rev, net_revenue=net, demand=dem, churn=chn))
        r = k.iloc[0]
        high = chn["overview"].loc[chn["overview"].risk_category == "High", "customers"].sum()
        ctx.kind = "mixed"
        ctx.say("**Actuals (database, real orders)**\n"
                f"- Total revenue {money(r.total_revenue)} from {count(r.total_orders)} orders by {count(r.total_customers)} customers "
                f"({r.first_order_date} to {r.last_order_date}); average order value {money(r.avg_order_value)}; average review {r.avg_review_score}.\n"
                f"- Net revenue {money(r.net_revenue)} after cancelled and unavailable orders (discounts, returns and refunds are not in the data).\n"
                f"- July 2018, the last complete month: {money(july.payment_total.iloc[0])}.\n"
                f"- Returning customers: {count(ret.returning_customers.iloc[0])} of {count(r.total_customers)}.",
                "**Forecasts (model estimates)**\n"
                f"- Revenue {rev['forecast_start']} to {rev['forecast_end']}: {money(rev['forecast_total'])} "
                f"({rev['growth_vs_previous_period_pct']:+.1f}% vs the previous 30 days, {rev['trend']}).\n"
                f"- Net revenue, same window: {money(net['forecast']['forecast_total'])}.\n"
                f"- Demand {dem['forecast_start']} to {dem['forecast_end']}: increasing {', '.join(dem['increasing']) or 'none'}; "
                f"decreasing {', '.join(dem['decreasing']) or 'none'}.\n"
                f"- Churn: {count(high)} of {count(chn['customers_scored'])} customers are High risk (model trained on real plus synthetic repeat orders).",
                "**Recommendations (rules applied to the agent outputs)**\n" + "\n".join(f"- {x}" for x in recs))
        ctx.table("Headline KPIs (actual)", k)
        ctx.table("Monthly revenue (actual)", self._q(ctx, "SELECT * FROM v_monthly_revenue ORDER BY purchase_month"))
        ctx.table("Daily revenue forecast, next 30 days (model output)", rev["forecast"].rename("predicted_value").rename_axis("date").reset_index())
        ctx.table("Demand forecast by category (model output)", dem["summary"])
        ctx.table("Churn risk overview (model output)", chn["overview"])

    # ------------------------------------------------------------ optional LLM mode
    @staticmethod
    def _llm_available() -> bool:
        if os.environ.get("CHATBOT_OFFLINE") or not os.environ.get("ANTHROPIC_API_KEY"):
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    def _llm(self, system: str, user: str, max_tokens: int = 2000) -> str | None:
        """One Claude call; returns the text or None on any failure (the caller then stays offline)."""
        try:
            import anthropic
            if self._llm_client is None:
                self._llm_client = anthropic.Anthropic()
            resp = self._llm_client.beta.messages.create(
                model=LLM_MODEL, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            if resp.stop_reason == "refusal":
                return None
            return "".join(b.text for b in resp.content if b.type == "text").strip() or None
        except Exception:  # noqa: BLE001 - any API problem falls back to offline behaviour
            return None

    def _schema_text(self) -> str:
        if "schema" not in self._meta:
            self._meta["schema"] = "\n".join(f"{t}({', '.join(cols)})" for t, cols in self.c.schema().items()
                                             if not t.startswith("stg_"))
        return self._meta["schema"]

    def h_llm_sql(self, question: str, ctx: _Ctx) -> bool:
        system = (
            "You translate a business question into ONE read-only SQLite SELECT statement for an e-commerce database. "
            "Return only the SQL inside a ```sql block, or the single word NONE if the question cannot be answered from the schema. "
            "Rules: use v_sales for real orders (one row per order; revenue = payment_total); never use v_orders_all or order_summary; "
            "when joining the orders table, always filter orders.is_synthetic = 0; exclude order_status 'canceled' and 'unavailable' for "
            "product or category sales; amounts are in Brazilian real; data is complete 2017-01-09 to 2018-08-19, 'last month' = 2018-07; "
            "there are no discounts, returns, refunds, costs, product names or customer names - answer NONE for those. "
            "Add LIMIT 200 to list queries.\n\nSchema (table(columns)):\n" + self._schema_text())
        text = self._llm(system, question, 1500)
        if not text or text.strip().upper().startswith("NONE"):
            return False
        m = re.search(r"```(?:sql)?\s*(.+?)```", text, re.S)
        sql = (m.group(1) if m else text).strip().rstrip(";")
        if re.search(r"\bv_orders_all\b|\border_summary\b", sql, re.I) or \
                (re.search(r"\b(from|join)\s+orders\b", sql, re.I) and not re.search(r"is_synthetic\s*=\s*0", sql, re.I)):
            return False
        try:
            df = self._q(ctx, sql, source="LLM-generated SELECT")
        except RequestError:
            return False
        df = df.head(MAX_ROWS)
        ctx.kind = "prediction" if re.search(r"forecast|churn_predictions", sql, re.I) else "historical"
        ctx.table("Query result", df)
        ctx.source("llm: query generated by Claude, executed read-only")
        context = df.head(50).to_csv(index=False)
        answer = self._llm(
            "Answer the question in 1-4 short markdown sentences using ONLY the query result given. Never use a number that is not in "
            "the result. Amounts are Brazilian real, formatted like R$ 1,234.56. If the result is empty, say no matching data was found. "
            + ("These are model predictions: say so." if ctx.kind == "prediction" else "These are historical records."),
            f"Question: {question}\n\nSQL: {sql}\n\nResult (CSV):\n{context}", 800)
        if answer and self._numbers_grounded(answer, context + " " + question):
            ctx.say(answer)
        else:
            ctx.say(f"Result of the generated query ({len(df)} rows) is in the table below.")
        return True

    def _llm_reword(self, question: str, ctx: _Ctx) -> None:
        passages = "\n\n".join(ctx.text)
        answer = self._llm("Answer the question in plain, direct markdown using ONLY the passages. Never introduce a number that is not "
                           "in the passages. Keep it under 150 words.", f"Question: {question}\n\nPassages:\n{passages}", 800)
        if answer and self._numbers_grounded(answer, passages + " " + question):
            ctx.text = [answer]
            ctx.source("llm: wording by Claude from the retrieved passages")

    @staticmethod
    def _numbers_grounded(answer: str, context: str) -> bool:
        """True when every number in the answer equals, or is a rounding of, a number in the context."""
        def nums(s):
            return [x.replace(",", "") for x in re.findall(r"\d[\d,]*(?:\.\d+)?", s)]
        ctx_vals = []
        for x in nums(context):
            try:
                ctx_vals.append(float(x))
            except ValueError:
                pass
        for x in nums(answer):
            try:
                v = float(x)
            except ValueError:
                return False
            dec = len(x.split(".")[1]) if "." in x else 0
            tol = 0.5 * 10 ** (-dec) + 1e-9
            if not any(abs(c - v) <= tol or abs(c * 100 - v) <= tol for c in ctx_vals):
                return False
        return True


# --------------------------------------------------------------------------- CLI
def _print_response(resp: dict) -> None:
    print(f"\n[{resp['kind']}]\n{resp['answer']}\n")
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 40):
        for t in resp["tables"]:
            print(f"--- {t['title']} ({len(t['data'])} rows)")
            print(t["data"].head(15).to_string(index=False))
            print()
    if resp["sources"]:
        print("Sources:", "; ".join(resp["sources"]))
    for s in resp["sql"]:
        print("SQL:", s)


def main(argv: list[str]) -> None:
    bot = Chatbot()
    if len(argv) > 1:
        _print_response(bot.ask(" ".join(argv[1:])))
        return
    print("E-commerce BI chatbot (offline mode)" if not bot.use_llm else "E-commerce BI chatbot (LLM mode)")
    print("Ask a question, or type 'exit'.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", "q"}:
            break
        _print_response(bot.ask(q))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv)
