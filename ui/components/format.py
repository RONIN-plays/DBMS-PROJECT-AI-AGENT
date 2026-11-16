"""Number and label formatting (Brazilian real, counts, percentages)."""
from __future__ import annotations

import math

import pandas as pd

NOT_AVAILABLE = "Not available in this dataset"


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def brl(value, decimals: int = 0) -> str:
    """R$ 1,234,567 (negative: -R$ 1,234)."""
    if _missing(value):
        return "n/a"
    if isinstance(value, str):
        return NOT_AVAILABLE
    sign = "-" if value < 0 else ""
    return f"{sign}R$ {abs(value):,.{decimals}f}"


def num(value, decimals: int = 0) -> str:
    return "n/a" if _missing(value) else f"{value:,.{decimals}f}"


def pct(value, decimals: int = 1, signed: bool = False) -> str:
    if _missing(value):
        return "n/a"
    return f"{value:+.{decimals}f}%" if signed else f"{value:.{decimals}f}%"


def md(text) -> str:
    """Escape '$' so Streamlit markdown does not read 'R$ 1 ... R$ 2' as LaTeX."""
    return "" if text is None else str(text).replace("\\$", "$").replace("$", "\\$")


def label(name: str) -> str:
    """'bed_bath_table' -> 'Bed bath table' (display only; the raw name is what the controller expects)."""
    text = str(name).replace("_", " ").strip()
    return text[:1].upper() + text[1:]


def tidy_table(df: pd.DataFrame, money: tuple = (), pcts: tuple = (), labels: tuple = (), rename: dict | None = None) -> pd.DataFrame:
    """Copy of df with money/percent columns as display strings and humanised headers."""
    out = df.copy()
    for col in money:
        if col in out:
            out[col] = out[col].map(brl)
    for col in pcts:
        if col in out:
            out[col] = out[col].map(lambda v: pct(v, 2))
    for col in labels:
        if col in out:
            out[col] = out[col].map(label)
    out.columns = [(rename or {}).get(c, label(c)) for c in out.columns]
    return out
