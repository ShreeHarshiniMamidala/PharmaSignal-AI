"""Statistical Signals — existing PharmaSignal AI result files, no model retraining."""
from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parent
if str(APP) not in sys.path:
    sys.path.insert(0,str(APP))
from signal_spectrum import apply_signal_spectrum, hero, note, chart_style, BLUE, VIOLET, PINK, AMBER

st.set_page_config(page_title="Statistical Signals | PharmaSignal AI",page_icon="📈",layout="wide",initial_sidebar_state="expanded")
from signal_spectrum import (
    apply_signal_spectrum,
    spectrum_header,
    style_spectrum_figure,
    COLORS,
)

apply_signal_spectrum()
hero("Statistical Signal Analysis","Investigate disproportionality measures, report support, and candidate drug–event associations using saved FAERS analysis results.","PHARMASIGNAL AI / STATISTICAL SIGNALS")

@st.cache_data(show_spinner=False)
def read_csv(path):
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def read_parquet(path):
    # Read pair-level summary only, not raw report-level data.
    return pd.read_parquet(path)

period=st.sidebar.selectbox("FAERS period",["2025","2026 Q1"],key="stats_period")
suffix="2025" if period=="2025" else "2026_Q1"
summary_path=ROOT/"results"/"signals"/f"signal_summary_{suffix}.csv"
full_path=ROOT/"data"/"processed"/"traditional_full"/f"full_traditional_signals_{suffix}.parquet"

if not summary_path.exists():
    st.error(f"Signal summary not found: {summary_path}")
    st.info("Copy the results/signals folder from your existing PharmaSignal AI repository.")
    st.stop()
summary=read_csv(summary_path)
if summary.empty:
    st.warning("The signal summary file contains no rows.")
    st.stop()
s=summary.iloc[0]

st.subheader("FAERS signal screening overview")
cols=st.columns(4)
metrics=[("Drug–event pairs","total_drug_event_pairs"),("Minimum-support pairs","pairs_meeting_min_support"),("ROR lower CI > 1","pairs_ror_lower_ci_gt_1"),("PRR > 2","pairs_prr_gt_2")]
for c,(label,key) in zip(cols,metrics):
    with c:
        val=pd.to_numeric(pd.Series([s.get(key)]),errors="coerce").iloc[0]
        st.metric(label,f"{int(val):,}" if pd.notna(val) else "—")
note("Screening counts are not confirmed adverse drug reactions. The ROR and PRR criteria shown are the criteria recorded in the source summary; they may overlap and are not mutually exclusive.")

st.subheader("Signal metric summaries")
left,right=st.columns([1,1],gap="large")
medians={"Median ROR":"median_ror","Median PRR":"median_prr","Median IC":"median_ic","Median pair reports":"median_pair_report_count"}
with left:
    for label,key in medians.items():
        v=pd.to_numeric(pd.Series([s.get(key)]),errors="coerce").iloc[0]
        st.metric(label,f"{v:,.3f}" if pd.notna(v) else "—")
with right:
    bars=pd.DataFrame([{"Screening group":name,"Pairs":pd.to_numeric(pd.Series([s.get(col)]),errors="coerce").iloc[0]} for name,col in [("Min support","pairs_meeting_min_support"),("ROR lower CI > 1","pairs_ror_lower_ci_gt_1"),("PRR > 2","pairs_prr_gt_2")]])
    fig=px.bar(bars,x="Screening group",y="Pairs",color="Screening group",color_discrete_sequence=[BLUE,VIOLET,PINK],text_auto=",.0f")
    fig.update_layout(showlegend=False,xaxis_title="",yaxis_title="Number of pairs")
    st.plotly_chart(chart_style(fig,370),use_container_width=True)

st.divider()
st.subheader("Explore individual drug–event signals")
if not full_path.exists():
    st.info("The complete pair-level signal parquet is not present. The summary above is fully available; individual signal exploration requires the saved processed/traditional_full parquet.")
    st.code(str(full_path))
    st.caption("This file may be excluded from GitHub because of its size. Copy it from your local project data folder; do not load the raw 16-million-row report-level dataset.")
else:
    try:
        with st.spinner("Loading saved pair-level signals…"):
            df=read_parquet(full_path)
    except Exception as exc:
        st.error(f"Unable to read the pair-level parquet: {exc}")
        st.stop()
    if "pair_report_count" not in df and "report_count" in df:
        df=df.rename(columns={"report_count":"pair_report_count"})
    required={"suspect_drug","adverse_event","pair_report_count"}
    if not required.issubset(df.columns):
        st.error(f"Missing columns in signal parquet: {sorted(required-set(df.columns))}")
        st.stop()
    df["pair_report_count"]=pd.to_numeric(df["pair_report_count"],errors="coerce")
    min_support=st.slider("Minimum pair-level reports",min_value=1,max_value=100,value=10,step=1)
    drugs=st.text_input("Drug name contains",placeholder="e.g., DUPIXENT").strip()
    events=st.text_input("Event term contains",placeholder="e.g., PRURITUS").strip()
    filtered=df.loc[df["pair_report_count"].ge(min_support)].copy()
    if drugs:
        filtered=filtered.loc[filtered["suspect_drug"].astype(str).str.contains(drugs,case=False,regex=False,na=False)]
    if events:
        filtered=filtered.loc[filtered["adverse_event"].astype(str).str.contains(events,case=False,regex=False,na=False)]
    st.caption(f"{len(filtered):,} pairs match the current filters. Charts display at most 1,000 highest-support pairs for responsiveness.")
    if filtered.empty:
        st.info("No drug–event pairs match the current filters.")
    else:
        sample=filtered.nlargest(min(len(filtered),1000),"pair_report_count").copy()
        numeric=[c for c in ["ror","prr","ic"] if c in sample.columns]
        for c in numeric:
            sample[c]=pd.to_numeric(sample[c],errors="coerce")
        if numeric:
            x_metric=st.selectbox("Signal-strength measure",numeric,format_func=lambda x:x.upper())
            plot=sample.loc[sample[x_metric].notna() & sample["pair_report_count"].gt(0)].copy()
            if not plot.empty:
                fig=px.scatter(plot,x="pair_report_count",y=x_metric,hover_name="suspect_drug",hover_data=["adverse_event"],color=x_metric,color_continuous_scale=[BLUE,VIOLET,PINK],log_x=True,labels={"pair_report_count":"Pair report count (log scale)",x_metric:x_metric.upper()})
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(chart_style(fig,480),use_container_width=True)
        view_cols=[c for c in ["suspect_drug","adverse_event","pair_report_count","ror","ror_ci_lower","ror_ci_upper","prr","ic","meets_min_reports","meets_ror_screen","meets_prr_screen","meets_ic_screen"] if c in filtered.columns]
        st.dataframe(filtered.nlargest(min(len(filtered),500),"pair_report_count")[view_cols],use_container_width=True,hide_index=True)
        st.download_button("Download filtered signal pairs",data=filtered[view_cols].to_csv(index=False).encode("utf-8"),file_name=f"statistical_signals_{suffix}.csv",mime="text/csv")

with st.expander("How to interpret ROR, PRR, and IC"):
    st.markdown("**ROR (Reporting Odds Ratio):** Relative reporting odds of a drug–event combination compared with other reports.\n\n**PRR (Proportional Reporting Ratio):** Ratio of the proportion of a reported event for the drug versus other drugs.\n\n**IC (Information Component):** Bayesian-style observed-versus-expected reporting measure.\n\nThese are disproportionality measures, not incidence rates or proof of causality.")
st.caption(f"PharmaSignal AI · Statistical Signals · FAERS {period} · Research use only")
