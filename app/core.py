"""Shared loaders, theme and UI helpers for the Streamlit app.

The app reads ONLY precomputed files in app/data/ (built by `make app-data`).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"
DOCS = Path(__file__).resolve().parents[1] / "docs"
REPO_URL = "https://github.com/Abhi237/supplier-risk-scoring"
PROJECT_TITLE = ("Health Supply Vendor Risk - Using empirical-Bayes shrinkage and multilevel models to score on-time "
                 "delivery risk on USAID / PEPFAR health-commodity order lines")
SOURCE_CITATION = ("U.S. Agency for International Development (USAID), PEPFAR Supply Chain Management System (SCMS) "
                   "Delivery History / Supply Chain Shipment Pricing Data, data.usaid.gov (Kaggle copy: "
                   "divyeshardeshana/supply-chain-shipment-pricing-data)")
MEMO_URL = f"{REPO_URL}/blob/main/docs/procurement_memo.pdf"

# ---- "Control tower" design language: dark slate base, teal/amber accents, RAG risk palette ----
BG, PANEL, PANEL2, LINE, TEXT, MUTED = "#0b1220", "#111a2e", "#16213a", "#24324f", "#e6edf7", "#9fb0c8"
TEAL, AMBER, RED, GREEN, CYAN, VIOLET, GREY = ("#2dd4bf", "#fbbf24", "#f87171", "#4ade80", "#38bdf8", "#a78bfa",
                                              "#94a3b8")
# legacy names used across views, remapped onto the control-tower palette
NAVY, ORANGE, PURPLE = CYAN, AMBER, VIOLET
COLORWAY = [TEAL, AMBER, CYAN, RED, VIOLET, GREEN, GREY]
MONO = "'JetBrains Mono', 'IBM Plex Mono', 'SFMono-Regular', Menlo, Consolas, monospace"
SANS = "'IBM Plex Sans', 'Inter', 'Segoe UI', Helvetica, Arial, sans-serif"

pio.templates["control_tower"] = go.layout.Template(layout=go.Layout(
    font=dict(family=SANS, size=13, color=TEXT), colorway=COLORWAY,
    paper_bgcolor=PANEL, plot_bgcolor=PANEL,
    xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE, tickfont=dict(family=MONO, color=MUTED)),
    yaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE, tickfont=dict(family=MONO, color=MUTED)),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.2, font=dict(color=MUTED)),
    margin=dict(l=55, r=20, t=56, b=50),
    title=dict(font=dict(size=13, family=MONO, color=TEXT), x=0.01),
    hoverlabel=dict(font_size=12, font_family=MONO, bgcolor=PANEL2, bordercolor=TEAL),
    coloraxis=dict(colorbar=dict(tickfont=dict(color=MUTED, family=MONO))),
))
pio.templates.default = "plotly_dark+control_tower"
SEQ = [[0, PANEL2], [1, TEAL]]                       # sequential scale for heatmaps
SEQ_RISK = [[0, "#14532d"], [0.5, AMBER], [1, RED]]  # green -> amber -> red
DIVERGE = [[0, CYAN], [0.5, PANEL2], [1, RED]]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap');
html, body, [data-testid="stAppViewContainer"] {{font-family: {SANS};}}
.block-container, [data-testid="stMainBlockContainer"] {{padding-top: 4.4rem !important; max-width: 1320px;}}
h1 {{font-family: {SANS}; font-weight: 600; font-size: 1.9rem !important; letter-spacing: -0.01em; margin-bottom: 0;}}
h2, h3 {{font-family: {MONO}; font-weight: 600; font-size: 1.0rem !important; text-transform: uppercase;
  letter-spacing: 0.08em; color: {TEAL} !important; border-bottom: 1px solid {LINE}; padding-bottom: 6px;}}
.eyebrow {{font-family: {MONO}; font-size: 0.74rem; letter-spacing: 0.14em; text-transform: uppercase; color: {TEAL};}}
.eyebrow span {{color: {MUTED};}}
.subtitle {{color: {MUTED}; font-size: 0.92rem; margin: 2px 0 8px 0;}}
.statusbar {{display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0 14px 0;}}
.chip {{font-family: {MONO}; font-size: 0.74rem; color: {TEXT}; background: {PANEL2}; border: 1px solid {LINE};
  border-radius: 3px; padding: 3px 9px;}}
.chip b {{color: {TEAL}; font-weight: 600;}}
.dot {{display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: middle;}}
/* native metrics restyled as flat status tiles: top accent rule, mono numerals */
[data-testid="stMetric"] {{background: {PANEL}; border: 1px solid {LINE}; border-top: 3px solid {TEAL};
  border-radius: 2px; padding: 12px 14px;}}
[data-testid="stMetricLabel"] p {{font-family: {MONO}; font-size: 0.7rem; color: {MUTED}; text-transform: uppercase;
  letter-spacing: 0.1em;}}
[data-testid="stMetricValue"] {{font-family: {MONO}; font-size: 1.55rem; font-weight: 600; color: {TEXT};}}
[data-testid="stMetricDelta"] {{font-family: {MONO}; font-size: 0.75rem;}}
/* custom RAG status tiles */
.tiles {{display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 10px; margin: 6px 0 12px 0;}}
.tile {{background: {PANEL}; border: 1px solid {LINE}; border-radius: 2px; padding: 12px 14px; position: relative;}}
.tile .lab {{font-family: {MONO}; font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; color: {MUTED};}}
.tile .val {{font-family: {MONO}; font-size: 1.6rem; font-weight: 600; color: {TEXT}; margin: 4px 0 2px 0;}}
.tile .sub {{font-size: 0.8rem; color: {MUTED};}}
.tile .led {{position: absolute; top: 12px; right: 12px; width: 10px; height: 10px; border-radius: 50%;}}
.tile.red {{border-left: 4px solid {RED};}} .tile.amber {{border-left: 4px solid {AMBER};}}
.tile.green {{border-left: 4px solid {GREEN};}} .tile.info {{border-left: 4px solid {CYAN};}}
.alert {{display: grid; grid-template-columns: 70px 1fr auto; gap: 10px; align-items: center; background: {PANEL};
  border: 1px solid {LINE}; border-left: 4px solid {RED}; padding: 8px 12px; margin-bottom: 6px; border-radius: 2px;}}
.alert.amber {{border-left-color: {AMBER};}}
.alert .sev {{font-family: {MONO}; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em;}}
.alert .id {{font-family: {MONO}; font-size: 0.85rem; color: {TEXT};}}
.alert .meta {{font-size: 0.78rem; color: {MUTED};}}
.alert .score {{font-family: {MONO}; font-size: 1.1rem; font-weight: 700;}}
.assumed {{background: rgba(251,191,36,0.08); border: 1px solid {AMBER}; border-radius: 2px; padding: 10px 14px;
  color: #fde68a;}}
.callout {{background: {PANEL}; border: 1px solid {LINE}; border-left: 4px solid {TEAL}; border-radius: 2px;
  padding: 12px 16px; color: {TEXT};}}
.strip {{display: grid; grid-template-columns: repeat(3, 1fr); gap: 0; border: 1px solid {LINE}; margin: 4px 0 10px 0;}}
.strip .part {{background: {PANEL}; padding: 12px 16px; border-right: 1px solid {LINE};}}
.strip .part:last-child {{border-right: none;}}
.strip .plab {{font-family: {MONO}; font-size: 0.7rem; letter-spacing: 0.14em; color: {TEAL}; font-weight: 700; margin-bottom: 4px;}}
.strip .ptxt {{font-size: 0.86rem; color: {TEXT}; line-height: 1.4;}}
.small {{font-size: 0.85rem; color: {MUTED};}}
[data-testid="stHeader"] {{background: {BG}; border-bottom: 1px solid {LINE};}}
</style>
"""

RAG = {"red": RED, "amber": AMBER, "green": GREEN, "info": CYAN}


def setup_page(title: str, subtitle: str | None = None, section: str = "") -> None:
    """Eyebrow breadcrumb + title + subtitle + status bar (data window, scope, provenance)."""
    st.markdown(CSS, unsafe_allow_html=True)
    s = summary()
    q = s["sql_summary"]
    st.markdown(f"<div class='eyebrow'>VENDOR CONTROL TOWER <span>/ {section or 'overview'}</span></div>",
                unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.markdown(f"<p class='subtitle'>{subtitle}</p>", unsafe_allow_html=True)
    oot = s["train_summary"]["oot_main"]
    chips = [f"<span class='dot' style='background:{GREEN}'></span>PIPELINE <b>OK</b>",
             f"SCHEDULED <b>{q['scheduled_min'][:4]} to {q['scheduled_max'][:4]}</b>",
             f"VENDORS <b>{q['ext_vendors']:,}</b>", f"LINES <b>{q['n_lines']:,}</b>",
             f"OOT SCORE DATE <b>{oot['score_end'][:10]}</b>",
             f"<span class='dot' style='background:{AMBER}'></span>BUYER = US GOVT HEALTH PROGRAMME · NOT US-DOMESTIC"]
    st.markdown("<div class='statusbar'>" + "".join(f"<span class='chip'>{c}</span>" for c in chips) + "</div>",
                unsafe_allow_html=True)


def strip(parts: list[tuple[str, str]]) -> None:
    """Compact labelled strip, e.g. PROBLEM / METHOD / DECISION."""
    html = "".join(f"<div class='part'><div class='plab'>{lab}</div><div class='ptxt'>{txt}</div></div>"
                   for lab, txt in parts)
    st.markdown(f"<div class='strip'>{html}</div>", unsafe_allow_html=True)


def tiles(items: list[tuple[str, str, str, str]]) -> None:
    """Render RAG status tiles: (label, value, sub-text, status in red/amber/green/info)."""
    html = "".join(f"<div class='tile {st_}'><span class='led' style='background:{RAG[st_]}'></span>"
                   f"<div class='lab'>{lab}</div><div class='val'>{val}</div><div class='sub'>{sub}</div></div>"
                   for lab, val, sub, st_ in items)
    st.markdown(f"<div class='tiles'>{html}</div>", unsafe_allow_html=True)


def assumed_banner(text: str) -> None:
    st.markdown(f"<div class='assumed'><b>ASSUMED INPUTS · SCENARIO ONLY.</b> {text}</div>",
                unsafe_allow_html=True)


def callout(text: str) -> None:
    st.markdown(f"<div class='callout'>{text}</div>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def summary() -> dict:
    return json.loads((DATA / "summary.json").read_text())


@st.cache_data(show_spinner=False)
def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


@st.cache_data(show_spinner=False)
def vendors() -> pd.DataFrame:
    return pd.read_parquet(DATA / "vendors.parquet")


@st.cache_data(show_spinner=False)
def oot_vendors() -> pd.DataFrame:
    return pd.read_parquet(DATA / "oot_vendors.parquet")


def usd(x: float) -> str:
    if abs(x) >= 1e9:
        return f"${x / 1e9:,.2f}bn"
    if abs(x) >= 1e6:
        return f"${x / 1e6:,.2f}m"
    if abs(x) >= 1e4:
        return f"${x / 1e3:,.0f}k"
    return f"${x:,.0f}"


def scope_note() -> None:
    st.caption("**Scope:** USAID / PEPFAR SCMS delivery history, one programme, scheduled 2006-2015. The buyer is a US "
               "government health-supply programme and deliveries go to PEPFAR countries (not US-domestic). On time = "
               "delivered-to-client date <= scheduled date; there is **no vendor hand-off timestamp**, so scores describe "
               "**recorded delivery performance for this programme, not vendor quality overall**, on limited samples.")


TIER_ORDER = ["High", "Elevated", "Moderate", "Low", "New / unscored"]
TIER_COLORS = {"High": RED, "Elevated": AMBER, "Moderate": CYAN, "Low": GREEN, "New / unscored": GREY}
