"""Data access for the dashboard: every controller call goes through here (cached, errors shown, never raised)."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controller.main import Controller, RequestError  # noqa: E402

# Only these dates are complete in the source extract; the forecasts start the day after.
DATA_START, DATA_END = "2017-01-09", "2018-08-19"


@st.cache_resource(show_spinner=False)
def get_controller() -> Controller:
    return Controller()


@st.cache_data(show_spinner=False)
def _run(feature: str, params: tuple) -> dict:
    return get_controller().run(feature, **dict(params))


@st.cache_data(show_spinner=False)
def _query(sql: str) -> object:
    return get_controller().query(sql)


def _call(fn, *args, quiet: bool = False):
    """Run a cached call; return (result, error_message). Shows st.error unless quiet."""
    try:
        return fn(*args), None
    except RequestError as exc:
        msg = str(exc)
    except FileNotFoundError as exc:
        msg = f"Data or a trained model is missing: {exc}"
    except Exception as exc:  # noqa: BLE001 - the UI must never show a traceback
        msg = f"The analysis failed ({type(exc).__name__}: {exc})"
    if not quiet:
        st.error(msg)
    return None, msg


def run(feature: str, quiet: bool = False, **params):
    """Cached `Controller.run`. Returns the result, or None after showing the error."""
    with st.spinner("Loading analysis..."):
        result, err = _call(_run, feature, tuple(sorted(params.items())), quiet=quiet)
    return (result, err) if quiet else result


def query(sql: str):
    """Cached read-only SQL through the controller. Returns a DataFrame or None."""
    result, _ = _call(_query, sql)
    return result


def daily_history(net: bool = False):
    """Actual daily revenue over the complete window (real orders only, from v_sales)."""
    status = " AND order_status NOT IN ('canceled', 'unavailable')" if net else ""
    df = query(f"SELECT purchase_date, SUM(payment_total) AS revenue FROM v_sales "
               f"WHERE purchase_date BETWEEN '{DATA_START}' AND '{DATA_END}'{status} GROUP BY purchase_date")
    if df is not None:
        import pandas as pd
        df = df.assign(purchase_date=pd.to_datetime(df.purchase_date)).set_index("purchase_date")["revenue"]
    return df


def mark_completeness(df, month_col: str = "purchase_month"):
    """Tag months: 'edge' (outside the complete window: tiny, incomplete extract), 'partial' (window starts
    9 Jan 2017 / ends 19 Aug 2018 inside the month) or 'complete'."""
    first, last = DATA_START[:7], DATA_END[:7]
    tag = df[month_col].map(lambda m: "edge" if m < first or m > last else "partial" if m in (first, last) else "complete")
    return df.assign(completeness=tag)


def default_recommendations():
    """Cross-agent recommendations, exactly as `full_analysis` builds them (same default inputs),
    but reusing the cached per-agent results so the overview does not pay for a second full run."""
    parts = dict(overview=run("overview"), revenue=run("revenue", horizon_days=30),
                 net_revenue=run("net_revenue", horizon_days=30), demand=run("demand", weeks=4),
                 churn=run("churn", top_n=20))
    if any(v is None for v in parts.values()):
        return None
    try:
        return Controller.recommendations(parts)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not build recommendations ({type(exc).__name__}: {exc})")
        return None
