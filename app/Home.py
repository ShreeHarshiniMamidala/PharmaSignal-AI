"""PharmaSignal AI — Signal Spectrum homepage. Copy to app/Home.py."""
from pathlib import Path
from html import escape
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP = Path(__file__).resolve().parent
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
from utils.data_loader import (
    get_faers_summary, get_top_pairs, search_drug, search_event,
    get_drug_names, get_event_names,
)

st.set_page_config(page_title="PharmaSignal AI | Signal Spectrum", page_icon="💊", layout="wide", initial_sidebar_state="expanded")

from signal_spectrum import (
    apply_signal_spectrum,
    spectrum_header,
    style_spectrum_figure,
    COLORS,
)

apply_signal_spectrum()

# Retain the shared stylesheet import used in your existing Home.py.
try:
    from styles import apply_styles
    apply_styles()
except ImportError:
    pass  # This page also includes its own complete Signal Spectrum CSS.

# Local Plotly styling avoids requiring a new utils/theme.py file.
def polish(fig, height=460):
    return fig.update_layout(height=height)

BLUE, VIOLET, PINK, AMBER = "#38BDF8", "#A78BFA", "#F472B6", "#FBBF24"
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{color-scheme:dark}
html,body,.stApp{font-family:Inter,Arial,sans-serif!important}
.stApp{background:radial-gradient(ellipse at 85% 0%,rgba(103,58,183,.18),transparent 36%),radial-gradient(ellipse at 0% 45%,rgba(14,165,233,.1),transparent 38%),#080F20!important;color:#EEF4FF!important}
[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:transparent!important}
.block-container{max-width:1560px!important;padding-top:1.8rem!important;padding-bottom:3rem!important}
h1,h2,h3,h4{color:#EEF4FF!important;letter-spacing:-.025em}
p,label,li{color:#DCE5F5}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0B1630,#091125)!important;border-right:1px solid #273858!important}
section[data-testid="stSidebar"] *{color:#DDE6FA}
section[data-testid="stSidebar"] a{border-radius:10px;transition:background .2s,transform .2s}
section[data-testid="stSidebar"] a:hover{background:#342A6055;transform:translateX(2px)}
div[data-testid="stMetric"]{background:linear-gradient(140deg,#172747,#101C34)!important;border:1px solid #304365!important;border-radius:16px!important;padding:21px!important;transition:transform .22s,border-color .22s}
div[data-testid="stMetric"]:hover{transform:translateY(-3px);border-color:#A78BFA!important}
[data-testid="stMetricValue"]{color:#F8FAFF!important;font-weight:800!important}
[data-testid="stMetricLabel"]{color:#A7B8D5!important}
div[data-testid="stPlotlyChart"],div[data-testid="stDataFrame"]{background:#101D36!important;border:1px solid #2A3B5C!important;border-radius:16px!important;padding:9px!important}
div[data-testid="stExpander"]{background:#101D36!important;border:1px solid #2A3B5C!important;border-radius:12px!important}
div[data-baseweb="select"]>div,div[data-baseweb="input"]>div{background:#172744!important;border:1px solid #3A4D70!important;color:#EEF4FF!important;border-radius:10px!important}
button[data-baseweb="tab"]{color:#A7B8D5!important}
button[data-baseweb="tab"][aria-selected="true"]{color:#C4B5FD!important}
.stButton>button,.stDownloadButton>button{border-radius:10px!important;border:1px solid #6D60BA!important;background:#252657!important;color:#F5F3FF!important}
.stButton>button:hover,.stDownloadButton>button:hover{background:#40358A!important;border-color:#A78BFA!important}
hr{border-color:#273858!important}
.ps-hero{background:radial-gradient(circle at 92% 0%,#61306455,transparent 37%),linear-gradient(110deg,#10243F,#181A42);border:1px solid #344269;border-radius:20px;padding:34px 40px;margin:8px 0 26px;animation:psEnter .6s ease both}
.ps-kicker{font-size:11px;font-weight:800;letter-spacing:.17em;color:#C4B5FD;margin-bottom:13px}
.ps-hero h1{font-size:clamp(31px,3.5vw,52px);line-height:1.13;margin:0 0 12px;background:linear-gradient(90deg,#fff,#70D5FF,#D0B4FF);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.ps-hero p{color:#B8C7E5;line-height:1.75;max-width:820px;margin:0}
.ps-chip{display:inline-block;margin-top:20px;padding:8px 14px;border-radius:30px;border:1px solid #615CA0;color:#DDD6FE;background:#2B285B66;font-size:12px}
.ps-section{font-size:21px;font-weight:750;color:#F4F6FF;margin:22px 0 3px}
.ps-sub{font-size:12px;color:#A2B2D0;margin-bottom:14px}
.ps-note{padding:15px 18px;border:1px solid #334469;border-left:3px solid #A78BFA;border-radius:12px;background:#142440;color:#C9D5EB;font-size:13px;line-height:1.7}
.ps-brand{font-weight:800;font-size:20px;color:#F8FAFC}.ps-brand span{color:#F472B6}
@keyframes psEnter{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
@media(max-width:700px){.ps-hero{padding:24px 21px}}
@media(prefers-reduced-motion:reduce){.ps-hero{animation:none}}
</style>
""", unsafe_allow_html=True)


def section(title, subtitle):
    st.markdown(f'<div class="ps-section">{escape(title)}</div><div class="ps-sub">{escape(subtitle)}</div>', unsafe_allow_html=True)


def chart_style(fig, height=440):
    fig = polish(fig, height=height)
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#E6EDFF", size=12), margin=dict(l=10, r=22, t=35, b=25), hoverlabel=dict(bgcolor="#1B2B49", font_color="white"))
    fig.update_xaxes(gridcolor="#273854", zeroline=False)
    fig.update_yaxes(gridcolor="#273854", zeroline=False)
    return fig


def nfmt(v):
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return "—"


@st.cache_data(show_spinner=False)
def summary_for(period):
    return get_faers_summary(period)


@st.cache_data(show_spinner=False)
def top_for(period, count):
    return get_top_pairs(period=period, n=count)


@st.cache_data(show_spinner=False)
def names_for(period, kind):
    return list(get_drug_names(period) if kind == "drug" else get_event_names(period))


@st.cache_data(show_spinner=False)
def search_for(period, name, kind):
    return search_drug(period, name) if kind == "drug" else search_event(period, name)


def result_chart(frame, name_col, title, color):
    if not {name_col, "report_count"}.issubset(frame.columns):
        st.info("The expected chart columns are not available.")
        return
    view = frame.copy()
    view["report_count"] = pd.to_numeric(view["report_count"], errors="coerce")
    view = view.dropna(subset=["report_count"]).nlargest(18, "report_count").sort_values("report_count")
    if view.empty:
        st.info("No report counts available.")
        return
    fig = go.Figure(go.Bar(x=view["report_count"], y=view[name_col], orientation="h", marker_color=color, hovertemplate="%{y}<br>Pair reports: %{x:,}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title="Pair-level report count", yaxis_title="")
    st.plotly_chart(chart_style(fig, max(400, len(view) * 29 + 110)), use_container_width=True)


# SIDEBAR
st.sidebar.markdown('<div class="ps-brand"><span>⌁</span> PharmaSignal AI</div>', unsafe_allow_html=True)
st.sidebar.caption("SIGNAL SPECTRUM · Research analytics")
period = st.sidebar.selectbox("FAERS reporting period", ["2025", "2026 Q1"])
st.sidebar.divider()
st.sidebar.markdown("**RESEARCH WORKSPACE**")
st.sidebar.caption("Use the sidebar page navigation for Model Comparison, Statistical Signals, and Priority Matrix.")
st.sidebar.divider()
st.sidebar.caption("FAERS reports describe suspected associations, not established causation.")

# HERO
st.markdown('<div class="ps-hero"><div class="ps-kicker">PHARMACOVIGILANCE / SCIENTIFIC INTELLIGENCE</div><h1>Discover signals.<br>Understand the evidence.</h1><p>Explore adverse-event reporting patterns, investigate suspect drugs and reported reactions, and connect descriptive evidence to your statistical and machine-learning research.</p><div class="ps-chip">● Exploratory FAERS analysis</div></div>', unsafe_allow_html=True)

try:
    summary = summary_for(period)
except Exception as exc:
    st.error(f"Unable to load the FAERS summary: {exc}")
    st.stop()

section("Dataset pulse", f"Overview of cleaned FAERS data · {period}")
cols = st.columns(4)
for col, title, key in zip(cols, ["Safety reports", "Suspect drugs", "Adverse-event terms", "Drug–event pairs"], ["reports", "drugs", "events", "pairs"]):
    col.metric(title, nfmt(summary.get(key)))

section("Signal landscape", "Reported drug–event pairs by frequency; this is not a disproportionality or causal plot")
top_n = st.slider("Number of pairs", min_value=5, max_value=30, value=15, step=5)
try:
    pairs = top_for(period, top_n).copy()
except Exception as exc:
    st.warning(f"Unable to load reported pairs: {exc}")
    pairs = pd.DataFrame()

if not pairs.empty and {"suspect_drug", "adverse_event", "report_count"}.issubset(pairs.columns):
    pairs["report_count"] = pd.to_numeric(pairs["report_count"], errors="coerce")
    pairs = pairs.dropna(subset=["report_count"])
    pairs["pair"] = pairs["suspect_drug"].astype(str) + " · " + pairs["adverse_event"].astype(str)
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        fig = px.scatter(pairs, x="report_count", y="pair", size="report_count", color="report_count", color_continuous_scale=[[0, BLUE], [.5, VIOLET], [.8, PINK], [1, AMBER]], hover_data=["suspect_drug", "adverse_event"])
        fig.update_traces(marker=dict(opacity=.9, line=dict(color="#D8CAFF", width=.7)))
        fig.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Reports", yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(chart_style(fig, 490), use_container_width=True)
    with right:
        st.markdown("#### Most frequently reported")
        small = pairs.nlargest(8, "report_count").sort_values("report_count")
        fig = go.Figure(go.Bar(x=small["report_count"], y=small["pair"], orientation="h", marker_color=VIOLET, hovertemplate="%{y}<br>Reports: %{x:,}<extra></extra>"))
        fig.update_layout(xaxis_title="Reports", yaxis_title="")
        st.plotly_chart(chart_style(fig, 460), use_container_width=True)
    with st.expander("Inspect and download the underlying pairs"):
        st.dataframe(pairs[["suspect_drug", "adverse_event", "report_count"]].sort_values("report_count", ascending=False), hide_index=True, use_container_width=True)
        st.download_button("Download pair data", pairs.to_csv(index=False).encode("utf-8"), file_name="pharmasignal_top_pairs.csv", mime="text/csv")
else:
    st.info("No pair-level reporting data available for this period.")

st.markdown('<div class="ps-note">Reporting frequency measures how often a drug–event pair appears in the dataset. It does not measure incidence or demonstrate causality. ROR, PRR, and IC require separate statistical result data.</div>', unsafe_allow_html=True)

section("Drug–event explorer", "Search reported associations using the existing FAERS data loader")
drug_tab, event_tab = st.tabs(["Explore by drug", "Explore by adverse event"])
with drug_tab:
    try:
        drug_names = names_for(period, "drug")
    except Exception as exc:
        st.warning(f"Unable to load drug names: {exc}")
        drug_names = []
    drug = st.selectbox("Suspect drug", ["Select a drug..."] + drug_names)
    if drug != "Select a drug...":
        try:
            found = search_for(period, drug, "drug")
            if found.empty:
                st.info("No results found.")
            else:
                if {"adverse_event", "report_count"}.issubset(found.columns):
                    a, b = st.columns(2)
                    a.metric("Event terms", nfmt(found["adverse_event"].nunique()))
                    b.metric("Sum of pair-level counts", nfmt(found["report_count"].sum()))
                    result_chart(found, "adverse_event", f"Events reported with {drug}", BLUE)
                st.dataframe(found, use_container_width=True, hide_index=True)
                st.download_button("Download drug results", found.to_csv(index=False).encode("utf-8"), file_name="drug_results.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"Drug search failed: {exc}")
with event_tab:
    try:
        event_names = names_for(period, "event")
    except Exception as exc:
        st.warning(f"Unable to load adverse-event names: {exc}")
        event_names = []
    event = st.selectbox("Adverse-event term", ["Select an event..."] + event_names)
    if event != "Select an event...":
        try:
            found = search_for(period, event, "event")
            if found.empty:
                st.info("No results found.")
            else:
                if {"suspect_drug", "report_count"}.issubset(found.columns):
                    a, b = st.columns(2)
                    a.metric("Suspect drugs", nfmt(found["suspect_drug"].nunique()))
                    b.metric("Sum of pair-level counts", nfmt(found["report_count"].sum()))
                    result_chart(found, "suspect_drug", f"Drugs reported with {event}", PINK)
                st.dataframe(found, use_container_width=True, hide_index=True)
                st.download_button("Download event results", found.to_csv(index=False).encode("utf-8"), file_name="event_results.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"Event search failed: {exc}")

section("Research modules", "Continue to specialized analysis pages using the sidebar")
a, b, c = st.columns(3)
a.info("**Statistical signals**\n\nExamine ROR, PRR, IC, and reporting support.")
b.info("**Model comparison**\n\nCompare machine-learning performance against OMOP controls.")
c.info("**Priority matrix**\n\nPrioritize reporting patterns using transparent screening rules.")
with st.expander("Scientific interpretation and limitations"):
    st.write("FAERS spontaneous reports are subject to reporting bias, incomplete information, and duplicate or follow-up reports. Pair-level sums may count the same safety report more than once across different combinations. Disproportionality measures identify reporting patterns, not causal effects. OMOP reference controls support limited model evaluation and do not replace clinical validation.")
st.caption(f"PharmaSignal AI · Signal Spectrum · {period} · Research use only")
