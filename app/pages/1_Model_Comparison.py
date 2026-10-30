
# ============================================================
# PHARMASIGNAL AI
# MACHINE LEARNING MODEL COMPARISON DASHBOARD
# ============================================================

from pathlib import Path
from html import escape

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from styles import apply_styles


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Model Comparison | PharmaSignal AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_styles()

# Same Signal Spectrum CSS as the approved homepage.
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


# ============================================================
# PATH CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RESULTS_FILE = (
    ROOT
    / "results"
    / "final_comparison"
    / "tables"
    / "final_method_comparison.csv"
)


# ============================================================
# COLORS AND METRICS
# ============================================================

MODEL_COLORS = {
    "Logistic Regression": "#38BDF8",
    "Random Forest": "#A78BFA",
    "Gradient Boosting": "#F472B6",
}

METRICS = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1 Score": "f1",
    "AUROC": "auroc",
    "PR-AUC": "pr_auc",
}


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_results():
    df = pd.read_csv(RESULTS_FILE)

    required = {
        "method",
        "method_group",
        "dataset",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = df[
        df["method_group"] == "Machine Learning"
    ].copy()

    for metric in METRICS.values():
        if metric not in df.columns:
            df[metric] = float("nan")

        df[metric] = pd.to_numeric(
            df[metric],
            errors="coerce",
        )

    if "n_controls" not in df.columns:
        df["n_controls"] = pd.NA

    return df


def style_figure(fig, height=430):
    """Match the Signal Spectrum dark Plotly theme."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial", color="#E6EDFF", size=12),
        title_font=dict(color="#F2F5FF", size=17),
        margin=dict(l=25, r=30, t=55, b=40),
        legend=dict(orientation="h", y=-0.25, x=0.5,
                    xanchor="center", title_text="", font=dict(color="#C2CEE7")),
        hoverlabel=dict(bgcolor="#1B2B49", font_color="#FFFFFF", font_size=13),
    )
    fig.update_xaxes(gridcolor="#273854", zeroline=False,
                     tickfont=dict(color="#A2B2D0"), title_font=dict(color="#B9C8E4"))
    fig.update_yaxes(gridcolor="#273854", zeroline=False,
                     tickfont=dict(color="#A2B2D0"), title_font=dict(color="#B9C8E4"))
    return fig


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    '<div class="ps-hero">'
    '<div class="ps-kicker">PHARMASIGNAL AI / MODEL EVALUATION</div>'
    '<h1>Compare models.<br>Measure the evidence.</h1>'
    '<p>Explore evaluation metrics, compare machine-learning approaches, '
    'and investigate the tradeoffs between classification and ranking '
    'performance on mapped OMOP reference controls.</p>'
    '<div class="ps-chip">● Machine learning · Evaluation studio</div>'
    '</div>', unsafe_allow_html=True,
)


# ============================================================
# CHECK DATA FILE
# ============================================================

if not RESULTS_FILE.exists():
    st.error(
        "The model evaluation results file was not found."
    )
    st.code(str(RESULTS_FILE))
    st.stop()

try:
    df = load_results()

except (OSError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Unable to load model results: {error}")
    st.stop()

if df.empty:
    st.warning(
        "No machine-learning evaluation results were found."
    )
    st.stop()


# ============================================================
# DATASET SELECTOR
# ============================================================

st.subheader("Evaluation Configuration")

periods = sorted(
    df["dataset"].dropna().astype(str).unique().tolist()
)

default_index = next(
    (
        i for i, value in enumerate(periods)
        if "2025" in value
    ),
    0,
)

selected_period = st.selectbox(
    "Select evaluation dataset",
    periods,
    index=default_index,
    help=(
        "Select the dataset or evaluation split "
        "used to calculate model performance."
    ),
)

filtered = df[
    df["dataset"].astype(str) == selected_period
].copy()

if filtered.empty:
    st.warning("No results found for this dataset.")
    st.stop()

st.caption(
    f"Showing {len(filtered)} model evaluation records "
    f"for {selected_period}."
)

st.divider()


# ============================================================
# MODEL LEADERBOARD
# ============================================================

st.subheader("🏆 Model Performance Leaderboard")

st.write(
    "A quick overview of model performance using "
    "F1 score as the primary ranking metric."
)

leaderboard = (
    filtered
    .dropna(subset=["f1"])
    .sort_values("f1", ascending=False)
    .reset_index(drop=True)
)

if not leaderboard.empty:

    top_models = leaderboard.head(3)

    columns = st.columns(len(top_models))

    for i, (_, row) in enumerate(top_models.iterrows()):

        with columns[i]:

            label = (
                "🥇 Highest F1"
                if i == 0
                else "🥈 Second Highest F1"
                if i == 1
                else "🥉 Third Highest F1"
            )

            st.metric(
                label=label,
                value=f"{row['f1']:.3f}",
                help=f"Model: {row['method']}",
            )

            st.markdown(
                f"**{row['method']}**"
            )

            if pd.notna(row["accuracy"]):
                st.caption(
                    f"Accuracy: {row['accuracy']:.2%}"
                )

            if pd.notna(row["auroc"]):
                st.caption(
                    f"AUROC: {row['auroc']:.3f}"
                )

else:
    st.info(
        "F1 scores are unavailable for this evaluation."
    )

st.divider()


# ============================================================
# INTERACTIVE METRIC RANKING
# ============================================================

st.subheader("📊 Interactive Model Ranking")

st.write(
    "Select an evaluation metric to see how the models "
    "rank against one another."
)

metric_name = st.selectbox(
    "Select performance metric",
    list(METRICS.keys()),
    index=3,
    key="ranking_metric",
)

metric_column = METRICS[metric_name]

ranking_df = (
    filtered[["method", metric_column]]
    .dropna(subset=[metric_column])
    .sort_values(metric_column, ascending=True)
)

if not ranking_df.empty:

    ranking_fig = go.Figure()

    ranking_fig.add_trace(
        go.Bar(
            x=ranking_df[metric_column],
            y=ranking_df["method"],
            orientation="h",
            marker=dict(
                color=[
                    MODEL_COLORS.get(model, "#A78BFA")
                    for model in ranking_df["method"]
                ],
                line=dict(width=0),
            ),
            text=ranking_df[metric_column],
            texttemplate="%{text:.3f}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"{metric_name}: "
                "%{x:.4f}"
                "<extra></extra>"
            ),
        )
    )

    ranking_fig.update_layout(
        title=f"{metric_name} Comparison",
        xaxis_title=metric_name,
        yaxis_title="",
        xaxis=dict(
            range=[0, 1.10],
        ),
        showlegend=False,
    )

    ranking_fig = style_figure(
        ranking_fig,
        height=350,
    )

    st.plotly_chart(
        ranking_fig,
        use_container_width=True,
    )

    best_row = ranking_df.iloc[-1]

    st.info(
        f"**Highest {metric_name}:** "
        f"{best_row['method']} "
        f"({best_row[metric_column]:.4f}). "
        "This ranking applies to the selected "
        "dataset and metric only."
    )

else:
    st.warning(
        f"No valid {metric_name} values are available."
    )

st.divider()


# ============================================================
# PERFORMANCE HEATMAP
# ============================================================

st.subheader("🔥 Model Performance Heatmap")

st.write(
    "The heatmap provides a compact view of model "
    "performance across evaluation metrics. "
    "Brighter spectral colors represent higher scores."
)

heatmap_metrics = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1 Score": "f1",
    "AUROC": "auroc",
    "PR-AUC": "pr_auc",
}

heatmap_df = (
    filtered
    .set_index("method")[
        list(heatmap_metrics.values())
    ]
    .rename(
        columns={
            value: key
            for key, value in heatmap_metrics.items()
        }
    )
)

heatmap_df = heatmap_df.dropna(
    axis=1,
    how="all",
)

if not heatmap_df.empty and len(heatmap_df.columns) > 0:

    heatmap_values = heatmap_df.to_numpy(dtype=float)

    heatmap_fig = go.Figure(
        go.Heatmap(
            z=heatmap_values,
            x=heatmap_df.columns.tolist(),
            y=heatmap_df.index.tolist(),
            zmin=0,
            zmax=1,
            colorscale=[
                [0.0, "#162342"],
                [0.35, "#3156A4"],
                [0.7, "#8062D5"],
                [1.0, "#F472B6"],
            ],
            text=heatmap_values,
            texttemplate="%{text:.3f}",
            textfont=dict(size=14),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "%{x}: %{z:.4f}"
                "<extra></extra>"
            ),
            colorbar=dict(
                title="Score",
                thickness=14,
                len=0.85,
            ),
            hoverongaps=False,
        )
    )

    heatmap_fig.update_layout(
        height=max(300, 95 * len(heatmap_df) + 90),
        xaxis=dict(
            side="top",
            showgrid=False,
        ),
        yaxis=dict(
            autorange="reversed",
            showgrid=False,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#E6EDFF",
            size=13,
        ),
        margin=dict(
            l=20,
            r=20,
            t=45,
            b=25,
        ),
    )

    heatmap_fig = style_figure(heatmap_fig, height=max(300, 95 * len(heatmap_df) + 90))
    st.plotly_chart(
        heatmap_fig,
        use_container_width=True,
    )

else:
    st.info("No metrics available for the heatmap.")

st.divider()


# ============================================================
# GROUPED PERFORMANCE COMPARISON
# ============================================================

st.subheader("📈 Side-by-Side Metric Comparison")

st.write(
    "Compare selected evaluation metrics across "
    "all available machine-learning models."
)

selected_metrics = st.multiselect(
    "Choose metrics",
    list(METRICS.keys()),
    default=[
        "Accuracy",
        "F1 Score",
        "AUROC",
        "PR-AUC",
    ],
    key="grouped_metrics",
)

if selected_metrics:

    selected_columns = [
        METRICS[metric]
        for metric in selected_metrics
    ]

    grouped_df = filtered.melt(
        id_vars="method",
        value_vars=selected_columns,
        var_name="Metric",
        value_name="Score",
    ).dropna(subset=["Score"])

    grouped_df["Metric"] = grouped_df["Metric"].map(
        {
            value: key
            for key, value in METRICS.items()
        }
    )

    grouped_fig = px.bar(
        grouped_df,
        x="Metric",
        y="Score",
        color="method",
        barmode="group",
        text_auto=".3f",
        color_discrete_map=MODEL_COLORS,
        title="Performance Across Selected Metrics",
    )

    grouped_fig.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    grouped_fig.update_layout(
        yaxis=dict(
            range=[0, 1.10],
            title="Evaluation Score",
        ),
        xaxis_title="",
        legend_title_text="",
    )

    grouped_fig = style_figure(
        grouped_fig,
        height=470,
    )

    st.plotly_chart(
        grouped_fig,
        use_container_width=True,
    )

else:
    st.info(
        "Select at least one metric to display the chart."
    )

st.divider()


# ============================================================
# MODEL INTERPRETATION
# ============================================================

st.subheader("🧠 Model Interpretation")

st.write(
    "Model evaluation involves several performance "
    "dimensions. A model with higher accuracy may "
    "not necessarily provide the best recall or "
    "ranking performance."
)

with st.expander(
    "Understanding the evaluation metrics",
    expanded=False,
):

    st.markdown(
        """
        **Accuracy:** Proportion of correctly classified
        reference controls.

        **Precision:** Proportion of predicted positive
        controls that are actually positive.

        **Recall:** Proportion of known positive
        controls identified by the model.

        **F1 Score:** Harmonic mean of precision
        and recall.

        **AUROC:** Measures discrimination across
        classification thresholds.

        **PR-AUC:** Summarizes the precision–recall
        relationship across thresholds.
        """
    )


if (
    "Random Forest" in filtered["method"].values
    and "Logistic Regression" in filtered["method"].values
):

    rf = filtered[
        filtered["method"] == "Random Forest"
    ].iloc[0]

    lr = filtered[
        filtered["method"] == "Logistic Regression"
    ].iloc[0]

    if pd.notna(rf["f1"]) and pd.notna(lr["f1"]):

        difference = rf["f1"] - lr["f1"]

        if difference > 0:
            st.success(
                "**Research observation:** Random Forest "
                f"achieved an F1 score {difference:.4f} "
                "higher than Logistic Regression on "
                "the selected evaluation dataset. "
                "This suggests a better precision–recall "
                "balance for these reference controls."
            )

        elif difference < 0:
            st.info(
                "**Research observation:** Logistic Regression "
                "achieved a higher F1 score than Random Forest "
                "on the selected evaluation dataset."
            )

        else:
            st.info(
                "Random Forest and Logistic Regression "
                "achieved equal F1 scores."
            )

st.divider()


# ============================================================
# DETAILED RESULTS TABLE
# ============================================================

st.subheader("📋 Detailed Model Evaluation Results")

display_columns = [
    "method",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auroc",
    "pr_auc",
    "n_controls",
]

table_df = filtered[
    display_columns
].copy()

table_df = table_df.rename(
    columns={
        "method": "Model",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        "auroc": "AUROC",
        "pr_auc": "PR-AUC",
        "n_controls": "OMOP Controls",
    }
)

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        col: st.column_config.NumberColumn(
            col,
            format="%.4f",
        )
        for col in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "AUROC",
            "PR-AUC",
        ]
    },
)

st.download_button(
    label="⬇ Download Model Comparison Results",
    data=filtered.to_csv(index=False),
    file_name="pharmasignal_model_comparison.csv",
    mime="text/csv",
    type="primary",
)

st.divider()


# ============================================================
# SCIENTIFIC INTERPRETATION NOTICE
# ============================================================

st.warning(
    "⚠️ **Scientific Interpretation**\n\n"
    "These results evaluate machine-learning models "
    "against mapped OMOP reference controls. "
    "Performance depends on the evaluation dataset, "
    "reference-label coverage, and validation design.\n\n"
    "Machine-learning predictions identify potential "
    "drug-safety reporting patterns. They do not "
    "establish causality or replace clinical and "
    "regulatory evaluation."
)

st.caption(
    "PharmaSignal AI | Machine Learning Model Comparison "
    f"| Evaluation Dataset: {selected_period}"
)
