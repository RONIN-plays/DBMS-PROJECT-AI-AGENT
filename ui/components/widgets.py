"""Reusable page pieces: KPI cards, actual/prediction badges, notes, tables, model evaluation block."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from charts import plots
from components.format import label, md

KIND = {  # badge text and colour; colours match the chart encoding (actual = blue, prediction = orange)
    "actual": ("Actual", "blue"),
    "prediction": ("Model prediction", "orange"),
    "synthetic": ("Prediction, synthetic data", "orange"),
}


def badge(kind: str) -> None:
    text, color = KIND[kind]
    st.badge(text, color=color)


def kpi(col, title: str, value: str, kind: str = "actual", note: str | None = None, delta: str | None = None,
        help: str | None = None) -> None:
    with col.container(border=True):
        badge(kind)
        st.metric(title, value, delta=delta, help=help)
        if note:
            st.caption(md(note))


def section_header(title: str, kind: str | None = None, caption: str | None = None) -> None:
    st.subheader(title)
    if kind:
        badge(kind)
    if caption:
        st.caption(md(caption))


def insights(lines: list[str] | None, title: str = "Insights") -> None:
    if not lines:
        return
    st.markdown(f"**{title}**")
    st.markdown("\n".join(f"- {line}" for line in map(md, lines)))


def table(df: pd.DataFrame, height: int | None = None, **column_config) -> None:
    kwargs = {"height": height} if height else {}
    st.dataframe(df, hide_index=True, width="stretch", column_config=column_config or None, **kwargs)


def chart(fig, data: pd.DataFrame | None = None, key: str | None = None) -> None:
    """Plotly chart plus its table view (every chart has a table twin)."""
    st.plotly_chart(fig, width="stretch", theme="streamlit", key=key, config={"displaylogo": False})
    if data is not None:
        with st.expander("Show data"):
            table(data)


def model_block(result: dict, metric: str, lower_is_better: bool, agent_label: str) -> None:
    """Ensemble weights and test metrics of the ensemble vs the single models (and naive baselines)."""
    ev = result.get("evaluation")
    weights = result.get("weights") or {}
    if ev is None or ev.empty:
        st.info("No evaluation results are available for this model.")
        return
    test_split = next((s for s in ev.split.unique() if str(s).startswith("test")), ev.split.iloc[-1])
    test = ev[ev.split == test_split].copy()
    st.caption(f"{agent_label}: weighted ensemble of five models. Metrics are measured on held-out data "
               f"({test_split}); 'naive' rows are simple baselines for comparison.")
    left, right = st.columns(2)
    with left:
        w = pd.DataFrame({"model": list(weights), "weight": list(weights.values())}).sort_values("weight", ascending=False)
        fig = plots.hbar(w.model.map(label), w.weight, x_title="Weight in the ensemble", value_format=".0%",
                         title="Ensemble weights")
        chart(fig, key=f"weights-{agent_label}")
    with right:
        test = test.sort_values(metric, ascending=lower_is_better)          # best model on top
        fig = plots.hbar(test.model.map(label), test[metric], x_title=metric.replace("_", "-"),
                         title=f"Test {metric.replace('_', '-')} ({'lower' if lower_is_better else 'higher'} is better)",
                         emphasis=list(test.model == "ensemble"), value_format=".1f" if lower_is_better else ".3f")
        chart(fig, key=f"metric-{agent_label}")
        st.caption("Blue bar = the ensemble actually used; grey = single models and baselines.")
    shown = ev.copy()
    shown.insert(2, "weight", shown.model.map(weights))
    shown["model"] = shown.model.map(label)
    num_cols = [c for c in shown.columns if c not in ("model", "split")]
    with st.expander("All validation and test metrics"):
        table(shown, **{c: st.column_config.NumberColumn(label(c) if c != "weight" else "Weight", format="%.3f")
                        for c in num_cols})
