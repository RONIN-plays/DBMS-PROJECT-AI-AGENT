"""F. Assistant: chat page for the RAG chatbot (chatbot.RAG.Chatbot().ask(question) -> dict)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components.format import md

TITLE = "Assistant"
EXAMPLES = [
    "What was the total revenue in 2017?",
    "Which categories will be in high demand next month?",
    "Why is net revenue lower than gross revenue?",
    "How many customers are at high risk of churning?",
]
KIND_BADGE = {
    "historical": ("Historical data", "blue"),
    "prediction": ("Model prediction", "orange"),
    "explanation": ("Explanation", "gray"),
    "mixed": ("Historical data and predictions", "violet"),
    "unavailable": ("Not available", "red"),
}


@st.cache_resource(show_spinner=False)
def _load_bot():
    from chatbot.RAG import Chatbot  # imported lazily: the module may not exist yet
    return Chatbot()


def get_bot():
    """Return (bot, error_message)."""
    try:
        return _load_bot(), None
    except ImportError as exc:
        return None, f"The chatbot module is not installed yet ({exc})."
    except Exception as exc:  # noqa: BLE001
        return None, f"The chatbot could not start ({type(exc).__name__}: {exc})."


def ask(bot, question: str) -> dict:
    try:
        resp = bot.ask(question)
    except Exception as exc:  # noqa: BLE001
        return {"answer": f"The assistant failed to answer ({type(exc).__name__}: {exc}).", "kind": "unavailable",
                "tables": [], "sql": [], "sources": []}
    if not isinstance(resp, dict):
        resp = {"answer": str(resp)}
    return {"answer": str(resp.get("answer", "")), "kind": resp.get("kind", "explanation"),
            "tables": resp.get("tables") or [], "sql": resp.get("sql") or [], "sources": resp.get("sources") or []}


def show_response(resp: dict) -> None:
    text, color = KIND_BADGE.get(resp.get("kind"), (str(resp.get("kind", "")).capitalize() or "Answer", "gray"))
    st.badge(text, color=color)
    st.markdown(md(resp.get("answer")) or "_No answer._")
    for i, tbl in enumerate(resp.get("tables", [])):
        df = tbl.get("data") if isinstance(tbl, dict) else None
        if isinstance(df, pd.DataFrame):
            if tbl.get("title"):
                st.markdown(f"**{tbl['title']}**")
            st.dataframe(df, hide_index=True, width="stretch")
    if resp.get("sql") or resp.get("sources"):
        with st.expander("SQL and sources"):
            for q in resp.get("sql", []):
                st.code(q, language="sql")
            if resp.get("sources"):
                st.markdown("\n".join(f"- {s}" for s in resp["sources"]))


def render() -> None:
    st.title(TITLE)
    st.caption("Ask about sales, revenue, demand or customers. Answers say whether they come from historical data "
               "or from model predictions.")
    bot, err = get_bot()
    if err:
        st.info(f"Chatbot not available. {err} The rest of the dashboard works without it.")

    history = st.session_state.setdefault("chat_history", [])
    cols = st.columns(len(EXAMPLES))
    clicked = None
    for col, q in zip(cols, EXAMPLES):
        if col.button(q, key=f"example-{q}", width="stretch", disabled=bot is None):
            clicked = q
    if history and st.button("Clear conversation", key="chat-clear"):
        history.clear()

    for msg in history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(md(msg["content"]))
            else:
                show_response(msg["content"])

    typed = st.chat_input("Ask a question", disabled=bot is None, key="chat-input")
    question = typed or clicked
    if question and bot is not None:
        history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(md(question))
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = ask(bot, question)
            show_response(resp)
        history.append({"role": "assistant", "content": resp})
