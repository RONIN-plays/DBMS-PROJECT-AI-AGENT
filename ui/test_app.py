"""Headless smoke test of the dashboard.  Run from the project root:  python -X utf8 -m unittest ui/test_app.py -v"""
from __future__ import annotations

import re
import sys
import types
import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent / "app.py")
TIMEOUT = 300


def problems(at: AppTest) -> list[str]:
    return [f"exception: {e.value}" for e in at.exception] + [f"st.error: {e.value}" for e in at.error]


def metrics(at: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


class DashboardTest(unittest.TestCase):
    def open(self, section: str) -> AppTest:
        at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
        self.assertEqual(problems(at), [])
        at.sidebar.radio(key="nav").set_value(section).run()
        self.assertEqual(problems(at), [], section)
        return at

    def test_overview(self):
        at = self.open("Business overview")
        m = metrics(at)
        self.assertEqual(m["Total revenue"], "R$ 16,008,872")
        self.assertEqual(m["Net revenue"], "R$ 15,739,137")
        self.assertEqual(m["Total orders"], "99,441")
        self.assertEqual(m["Total customers"], "96,096")
        self.assertEqual(m["Returning customers"], "2,997")
        self.assertEqual(m["Total sales (units)"], "112,650")
        self.assertGreaterEqual(len(at.markdown), 1)
        self.assertTrue(any("Revenue: the next 30 days" in md.value for md in at.markdown), "recommendations missing")
        at.toggle(key="ov-edges").set_value(True).run()
        self.assertEqual(problems(at), [])

    def test_revenue(self):
        at = self.open("Revenue")
        self.assertIn("Predicted revenue, next 30 days", metrics(at))
        at.slider(key="rev-horizon").set_value(90).run()
        at.radio(key="rev-window").set_value("All complete history").run()
        self.assertEqual(problems(at), [])
        self.assertIn("Predicted revenue, next 90 days", metrics(at))
        at.slider(key="rev-horizon").set_value(7).run()
        self.assertEqual(problems(at), [])

    def test_customers(self):
        at = self.open("Customers")
        self.assertEqual(metrics(at)["Returning customers"], "2,997 of 96,096")
        self.assertTrue(any("synthetic" in w.value for w in at.warning), "synthetic-data note missing")
        at.slider(key="cust-topn").set_value(50).run()
        self.assertEqual(problems(at), [])
        at.text_input(key="cust-id").input("not-a-real-customer")
        at.button[0].click().run()          # the form submit button
        self.assertEqual(problems(at), [])
        self.assertTrue(any("No scored customer" in w.value for w in at.warning), "graceful not-found missing")
        at.text_input(key="cust-id").input("0753264f31df82553453f8f93f65306b")
        at.button[0].click().run()
        self.assertEqual(problems(at), [])
        self.assertIn("Churn probability", metrics(at))

    def test_demand(self):
        at = self.open("Product demand")
        at.slider(key="dem-weeks").set_value(8).run()
        at.selectbox(key="dem-chartcat").set_value("toys").run()
        self.assertEqual(problems(at), [])
        at.selectbox(key="dem-cat").set_value("sports_leisure").run()
        self.assertEqual(problems(at), [])
        at.slider(key="dem-weeks").set_value(12).run()
        self.assertEqual(problems(at), [])

    def test_net_revenue(self):
        at = self.open("Net revenue")
        m = metrics(at)
        self.assertEqual(m["Gross revenue"], "R$ 16,008,872")
        self.assertEqual(m["Net revenue"], "R$ 15,739,137")
        text = " ".join(str(df.value.to_string()) for df in at.dataframe)
        self.assertEqual(text.count("Not available in this dataset"), 3, "discounts/returns/refunds must show as not available")
        unescaped = [el.value for el in list(at.markdown) + list(at.caption) if re.search(r"(?<!\\)\$", el.value)]
        self.assertEqual(unescaped, [], "'R$' in markdown must be escaped or it renders as LaTeX")
        at.select_slider(key="net-period").set_value(("2018-01", "2018-06")).run()
        at.slider(key="net-horizon").set_value(60).run()
        self.assertEqual(problems(at), [])
        self.assertIn("2018-01 to 2018-06", " ".join(md.value for md in at.markdown))
        at.toggle(key="net-edges").set_value(True).run()
        self.assertEqual(problems(at), [])

    def test_assistant(self):
        at = self.open("Assistant")
        try:
            import chatbot.RAG  # noqa: F401
            real = True
        except ImportError:
            real = False
        if real:
            at.chat_input(key="chat-input").set_value("What was the total revenue?").run()
            self.assertEqual([str(e.value) for e in at.exception], [])
            self.assertEqual(len(at.chat_message), 2, "question and answer should both be shown")
        else:
            self.assertTrue(any("Chatbot not available" in i.value for i in at.info))

    def test_assistant_contract_with_stub(self):
        """Exercise the chat page against the agreed contract using a stand-in Chatbot (real module swapped out)."""
        import streamlit as st
        saved = {k: sys.modules.pop(k) for k in ("chatbot", "chatbot.RAG") if k in sys.modules}
        st.cache_resource.clear()

        class Chatbot:
            def ask(self, question):
                return {"answer": f"**Stub** answer to: {question}", "kind": "historical",
                        "tables": [{"title": "Example", "data": pd.DataFrame({"a": [1, 2]})}],
                        "sql": ["SELECT 1"], "sources": ["v_sales"]}

        pkg, mod = types.ModuleType("chatbot"), types.ModuleType("chatbot.RAG")
        mod.Chatbot = Chatbot
        pkg.RAG = mod
        sys.modules.update({"chatbot": pkg, "chatbot.RAG": mod})
        try:
            at = self.open("Assistant")
            at.button(key="example-What was the total revenue in 2017?").click().run()
            self.assertEqual(problems(at), [])
            at.chat_input(key="chat-input").set_value("Hello").run()
            self.assertEqual(problems(at), [])
            answers = [md.value for md in at.markdown if "Stub" in md.value]
            self.assertEqual(len(answers), 2)
            self.assertGreaterEqual(len(at.dataframe), 2)
        finally:
            sys.modules.pop("chatbot", None)
            sys.modules.pop("chatbot.RAG", None)
            sys.modules.update(saved)
            st.cache_resource.clear()


if __name__ == "__main__":
    unittest.main(verbosity=2)
