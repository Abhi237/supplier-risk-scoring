"""Supplier Reliability & Procurement Risk Scoring - multipage portfolio app (entrypoint).

Run locally:  streamlit run app/streamlit_app.py
Reads only precomputed artifacts in app/data/ (refresh with `make app-data`).
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402,F401  (registers the control-tower Plotly template)

st.set_page_config(page_title="Supplier Risk | Procurement Reliability Analytics", page_icon="🛰️", layout="wide",
                   initial_sidebar_state="collapsed")

V = Path(__file__).resolve().parent / "views"
pages = {
    "Story": [
        st.Page(V / "overview.py", title="Executive summary", icon=":material/insights:", default=True),
        st.Page(V / "data_quality.py", title="Data & quality", icon=":material/fact_check:"),
        st.Page(V / "delivery.py", title="Delivery performance", icon=":material/local_shipping:"),
    ],
    "Score": [
        st.Page(V / "scorecards.py", title="Seller scorecards", icon=":material/badge:"),
        st.Page(V / "validation.py", title="Risk model & validation", icon=":material/verified:"),
        st.Page(V / "archetypes.py", title="Seller archetypes", icon=":material/hub:"),
    ],
    "Decide": [
        st.Page(V / "exposure.py", title="Exposure & scenario simulator", icon=":material/tune:"),
        st.Page(V / "methodology.py", title="Methodology & limitations", icon=":material/menu_book:"),
    ],
}
nav = st.navigation(pages, position="top")
nav.run()
