"""E-commerce business intelligence dashboard.  Run from the project root:  streamlit run ui/app.py"""
from __future__ import annotations

import sys
from pathlib import Path

UI = Path(__file__).resolve().parent
for p in (UI, UI.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st  # noqa: E402

from sections import assistant, customers, demand, net_revenue, overview, revenue  # noqa: E402

SECTIONS = {m.TITLE: m for m in (overview, revenue, customers, demand, net_revenue, assistant)}

st.set_page_config(page_title="E-commerce BI", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
  .block-container {padding-top: 2rem; max-width: 1400px;}
  [data-testid="stMetricValue"] {font-size: 1.6rem;}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### E-commerce BI")
    st.caption("Olist Brazilian e-commerce, 2016-2018")
    choice = st.radio("Section", list(SECTIONS), key="nav", label_visibility="collapsed")
    st.divider()
    st.markdown(":blue-badge[Actual] historical figures from real orders")
    st.markdown(":orange-badge[Model prediction] forecasts and churn scores")
    st.caption("Charts: solid blue line = actual, dashed orange line on a shaded band = forecast. "
               "Currency: Brazilian real (R\\$).")

SECTIONS[choice].render()
