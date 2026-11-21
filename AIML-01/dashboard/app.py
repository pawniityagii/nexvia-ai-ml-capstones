"""Analyst dashboard (Streamlit).

Reads the case store the scoring service writes to and gives an analyst the
review loop the brief asks for: alerts ranked by risk, case status, and a
feedback label that becomes future training signal. A separate tab shows the
largest suspicious fraud rings from the graph.

Run (after a demo/stream has populated cases):
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.serving import case_store          # noqa: E402

st.set_page_config(page_title="Fraud Analyst Console", layout="wide")


@st.cache_resource
def _conn():
    return case_store.connect()


conn = _conn()
st.title("Real-Time Fraud - Analyst Console")
st.caption("Education/portfolio project. Not a production or clinical fraud system.")

s = case_store.stats(conn)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions scored", f"{s['n']:,}")
c2.metric("Flagged", f"{s['flagged']:,}")
c3.metric("Open cases", f"{s['open_cases']:,}")
c4.metric("Confirmed fraud", f"{s['confirmed']:,}")

tab_queue, tab_rings = st.tabs(["Case queue", "Fraud rings"])

with tab_queue:
    colf1, colf2 = st.columns([1, 1])
    status = colf1.selectbox("Status", ["open", "any", "confirmed_fraud", "cleared"])
    limit = colf2.slider("Rows", 10, 300, 50, step=10)
    rows = case_store.list_cases(conn, status=None if status == "any" else status,
                                 limit=limit)
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No cases yet. Run: python scripts/replay_demo.py")
    else:
        df["score"] = df["score"].round(4)
        st.dataframe(
            df[["transaction_id", "score", "decision", "amount", "status",
                "feedback", "card_uid"]],
            use_container_width=True, hide_index=True)

        st.subheader("Review a case")
        tid = st.selectbox("Transaction", df["transaction_id"].tolist())
        a1, a2, a3 = st.columns(3)
        if a1.button("Confirm fraud"):
            case_store.set_status(conn, tid, "confirmed_fraud")
            case_store.set_feedback(conn, tid, 1)
            st.rerun()
        if a2.button("Clear (legit)"):
            case_store.set_status(conn, tid, "cleared")
            case_store.set_feedback(conn, tid, 0)
            st.rerun()
        if a3.button("Reopen"):
            case_store.set_status(conn, tid, "open")
            st.rerun()

with tab_rings:
    st.write("Largest connected components sharing a card, device or email.")
    if st.button("Compute fraud rings (scans the dataset)"):
        from src.common import data_loader
        from src.features import feature_engineering as fe
        from src.models import graph_builder
        with st.spinner("Building graph..."):
            feats = fe.engineer(data_loader.load_raw())
            rings = graph_builder.ring_summary(feats, top=20)
        st.dataframe(rings, use_container_width=True, hide_index=True)
