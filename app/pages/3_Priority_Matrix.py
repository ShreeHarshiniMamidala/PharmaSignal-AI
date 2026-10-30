"""Priority Matrix — existing top-50 research priority data or full pair-level results."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

APP=Path(__file__).resolve().parents[1]
ROOT=APP.parent
if str(APP) not in sys.path:
    sys.path.insert(0,str(APP))
from signal_spectrum import apply_signal_spectrum,hero,note,chart_style,BLUE,VIOLET,PINK,AMBER

st.set_page_config(page_title="Priority Matrix | PharmaSignal AI",page_icon="🎯",layout="wide",initial_sidebar_state="expanded")
from signal_spectrum import (
    apply_signal_spectrum,
    spectrum_header,
    style_spectrum_figure,
    COLORS,
)

apply_signal_spectrum()
hero("Signal Priority Matrix","Prioritize drug–event reporting patterns by statistical strength and report support. Explore transparent screening thresholds without treating candidate signals as clinical risk scores.","PHARMASIGNAL AI / PRIORITY MATRIX")

@st.cache_data(show_spinner=False)
def read_csv(path):
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def read_parquet(path):
    return pd.read_parquet(path)

period=st.sidebar.selectbox("FAERS period",["2025","2026 Q1"],key="priority_period")
suffix="2025" if period=="2025" else "2026_Q1"
full_path=ROOT/"data"/"processed"/"traditional_full"/f"full_traditional_signals_{suffix}.parquet"
poster_path=ROOT/"src"/"Preliminary"/"results"/"tables"/"priority_matrix_top50.csv"

if full_path.exists():
    source="Full pair-level signal table"
    try:
        with st.spinner("Loading saved signal table…"):
            df=read_parquet(full_path)
    except Exception as exc:
        st.error(f"Unable to load signal parquet: {exc}")
        st.stop()
elif period=="2025" and poster_path.exists():
    source="Preliminary poster top-50 (2025 only)"
    df=read_csv(poster_path)
    note("Showing the existing PRELIMINARY 2025 poster top-50 selection, not the complete final FAERS signal universe. Quadrant assignments below are recomputed using the controls you choose, not the poster's original priority labels.")
else:
    st.warning("No pair-level priority data available for the selected period.")
    st.info("Add the full_traditional_signals parquet to data/processed/traditional_full/. For 2025 only, the page can alternatively display your existing preliminary priority_matrix_top50.csv.")
    st.code(str(full_path))
    st.stop()

if "pair_report_count" not in df and "report_count" in df:
    df=df.rename(columns={"report_count":"pair_report_count"})
required={"suspect_drug","adverse_event","pair_report_count","ror"}
if not required.issubset(df.columns):
    st.error(f"Priority data is missing: {sorted(required-set(df.columns))}")
    st.stop()

st.caption(f"Source: {source} · Period: {period} · {len(df):,} source rows")
for col in ["pair_report_count","ror","prr","ic","ror_ci_lower"]:
    if col in df:
        df[col]=pd.to_numeric(df[col],errors="coerce")

st.subheader("Configure the priority matrix")
c1,c2,c3=st.columns(3)
with c1:
    metric=st.selectbox("Signal-strength axis",[c for c in ["ror","prr","ic"] if c in df],format_func=lambda c:c.upper())
with c2:
    support_cut=st.number_input("High-support threshold (reports)",min_value=1,value=100,step=10)
with c3:
    strength_default={"ror":2.0,"prr":2.0,"ic":1.0}[metric]
    strength_cut=st.number_input(f"High {metric.upper()} threshold",min_value=0.0,value=strength_default,step=0.5)

f1,f2=st.columns(2)
with f1:
    drug_filter=st.text_input("Filter suspect drug",placeholder="Optional drug-name fragment").strip()
with f2:
    event_filter=st.text_input("Filter adverse event",placeholder="Optional event-name fragment").strip()

frame=df.loc[df[metric].notna() & df["pair_report_count"].gt(0)].copy()
if drug_filter:
    frame=frame.loc[frame["suspect_drug"].astype(str).str.contains(drug_filter,case=False,regex=False,na=False)]
if event_filter:
    frame=frame.loc[frame["adverse_event"].astype(str).str.contains(event_filter,case=False,regex=False,na=False)]
if frame.empty:
    st.info("No drug–event pairs match the selected filters.")
    st.stop()

high_support=frame["pair_report_count"].ge(support_cut)
high_strength=frame[metric].ge(strength_cut)
frame["priority_region"]=np.select(
    [high_support & high_strength,high_support & ~high_strength,~high_support & high_strength],
    ["High support / high strength","High support / lower strength","Lower support / high strength"],
    default="Lower support / lower strength"
)
priority_order=["High support / high strength","High support / lower strength","Lower support / high strength","Lower support / lower strength"]
colors={priority_order[0]:PINK,priority_order[1]:BLUE,priority_order[2]:AMBER,priority_order[3]:VIOLET}

cols=st.columns(4)
for col,name in zip(cols,priority_order):
    col.metric(name,f"{int(frame['priority_region'].eq(name).sum()):,}")

st.subheader("Interactive support × signal-strength map")
max_points=st.slider("Maximum plotted points",min_value=100,max_value=5000,value=1000,step=100)
# Deterministic top-support selection, not a random or statistically representative sample.
plot=frame.nlargest(min(max_points,len(frame)),"pair_report_count").copy()
fig=px.scatter(plot,x="pair_report_count",y=metric,color="priority_region",color_discrete_map=colors,category_orders={"priority_region":priority_order},hover_name="suspect_drug",hover_data={"adverse_event":True,"pair_report_count":":,","priority_region":False},log_x=True,labels={"pair_report_count":"Pair-level report count (log scale)",metric:metric.upper(),"priority_region":"Quadrant"},opacity=.75)
fig.add_hline(y=strength_cut,line_dash="dash",line_color="#C4B5FD",annotation_text=f"{metric.upper()} threshold")
fig.add_vline(x=support_cut,line_dash="dash",line_color="#7DD3FC",annotation_text="Support threshold")
fig.update_traces(marker=dict(size=9,line=dict(width=.5,color="#DDE7FA")))
st.plotly_chart(chart_style(fig,560),use_container_width=True)
st.caption(f"Plot shows {len(plot):,} highest-support pairs from {len(frame):,} filtered rows. The quadrant counts above use all filtered rows, not just the plotted subset.")
note("Priority quadrants are exploratory rules based on the selected thresholds. High disproportionality and reporting volume are reasons for investigation, not evidence of causation, incidence, severity, or clinical impact.")

st.subheader("Review candidate signals")
region=st.selectbox("Show quadrant",["All regions"]+priority_order)
selected=frame if region=="All regions" else frame.loc[frame["priority_region"].eq(region)]
selected=selected.sort_values(["pair_report_count",metric],ascending=False)
show_cols=[c for c in ["suspect_drug","adverse_event","pair_report_count","ror","ror_ci_lower","ror_ci_upper","prr","ic","priority_region"] if c in selected]
st.caption(f"{len(selected):,} matching candidates; table preview shows up to 500 rows.")
st.dataframe(selected[show_cols].head(500),use_container_width=True,hide_index=True)
st.download_button("Download filtered priority matrix",data=selected[show_cols].to_csv(index=False).encode("utf-8"),file_name=f"priority_matrix_{suffix}.csv",mime="text/csv")
with st.expander("Priority matrix methodology and limitations"):
    st.markdown("The horizontal axis is the number of FAERS reports containing a suspect-drug/event pair. The vertical axis is the selected ROR, PRR, or IC measure. The dashed lines are **user-configurable exploration thresholds**. The resulting quadrants are not a validated clinical risk classification. The 2025 preliminary top-50 file is a preselected subset and cannot be used to infer counts across all FAERS pairs.\n\nFor a full-universe analysis, provide the saved `full_traditional_signals` parquet for the selected period.")
st.caption(f"PharmaSignal AI · Priority Matrix · FAERS {period} · Research use only")
