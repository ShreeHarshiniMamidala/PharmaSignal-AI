# ============================================================
# PHARMASIGNAL AI
# HOME PAGE / FAERS EXPLORER
# ============================================================

import streamlit as st
import plotly.express as px

from utils.data_loader import (
    get_faers_summary,
    get_top_pairs,
    search_drug,
    search_event,
    get_drug_names,
    get_event_names,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PharmaSignal AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

/* ---------------------------------------------------------
   PAGE
--------------------------------------------------------- */

.stApp {
    background:
        radial-gradient(
            circle at 5% 5%,
            rgba(20, 184, 166, 0.10),
            transparent 24%
        ),
        radial-gradient(
            circle at 95% 5%,
            rgba(99, 102, 241, 0.08),
            transparent 24%
        ),
        #f8fafc;
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 4rem;
    max-width: 1450px;
}


/* ---------------------------------------------------------
   HEADINGS
--------------------------------------------------------- */

h1 {
    color: #0f172a;
    font-weight: 800 !important;
    letter-spacing: -0.8px;
}

h2, h3 {
    color: #0f172a;
}


/* ---------------------------------------------------------
   HERO
--------------------------------------------------------- */

.pharma-hero {
    position: relative;
    overflow: hidden;

    padding: 42px 46px;

    border-radius: 28px;

    background:
        linear-gradient(
            120deg,
            #0f766e 0%,
            #0891b2 48%,
            #4f46e5 100%
        );

    box-shadow:
        0 18px 45px rgba(15, 118, 110, 0.22);

    margin-bottom: 32px;

    animation: heroAnimation 0.7s ease-out;
}

.pharma-hero-title {
    font-size: 50px;
    line-height: 1.1;
    font-weight: 800;
    color: white;
    margin-bottom: 8px;
}

.pharma-hero-subtitle {
    font-size: 21px;
    font-weight: 600;
    color: #ccfbf1;
    margin-bottom: 18px;
}

.pharma-hero-text {
    font-size: 16px;
    line-height: 1.7;
    color: #f0fdfa;
    max-width: 900px;
}

.pharma-status {
    display: inline-block;
    margin-top: 22px;

    padding: 8px 15px;

    border-radius: 25px;

    background: rgba(255,255,255,0.15);

    border: 1px solid rgba(255,255,255,0.25);

    color: white;

    font-size: 14px;
    font-weight: 600;
}

@keyframes heroAnimation {

    from {
        opacity: 0;
        transform: translateY(-12px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}


/* ---------------------------------------------------------
   METRIC CARDS
--------------------------------------------------------- */

div[data-testid="stMetric"] {

    background: white;

    border: 1px solid #e2e8f0;

    border-radius: 18px;

    padding: 22px;

    box-shadow:
        0 6px 18px rgba(15,23,42,0.06);

    position: relative;

    overflow: hidden;

    transition:
        transform 0.25s ease,
        box-shadow 0.25s ease;
}

div[data-testid="stMetric"]::before {

    content: "";

    position: absolute;

    top: 0;
    left: 0;

    width: 100%;
    height: 5px;

    background:
        linear-gradient(
            90deg,
            #14b8a6,
            #06b6d4,
            #6366f1
        );
}

div[data-testid="stMetric"]:hover {

    transform: translateY(-5px);

    box-shadow:
        0 14px 30px rgba(15,23,42,0.12);
}

div[data-testid="stMetricValue"] {

    color: #0f766e;

    font-weight: 800;
}


/* ---------------------------------------------------------
   INFO / ALERTS
--------------------------------------------------------- */

div[data-testid="stAlert"] {

    border-radius: 14px;

    box-shadow:
        0 3px 12px rgba(15,23,42,0.04);
}


/* ---------------------------------------------------------
   PLOTLY CHARTS
--------------------------------------------------------- */

div[data-testid="stPlotlyChart"] {

    background: white;

    border: 1px solid #e2e8f0;

    border-radius: 20px;

    padding: 10px;

    box-shadow:
        0 6px 18px rgba(15,23,42,0.05);
}


/* ---------------------------------------------------------
   TABS
--------------------------------------------------------- */

button[data-baseweb="tab"] {

    font-weight: 650;

    border-radius: 10px;

    padding-left: 18px;
    padding-right: 18px;
}

button[data-baseweb="tab"]:hover {

    background: #ecfeff;

    color: #0f766e;
}


/* ---------------------------------------------------------
   SELECT BOX
--------------------------------------------------------- */

div[data-baseweb="select"] > div {

    border-radius: 12px !important;

    transition: 0.2s ease;
}

div[data-baseweb="select"] > div:hover {

    border-color: #14b8a6 !important;
}


/* ---------------------------------------------------------
   DATA TABLE
--------------------------------------------------------- */

div[data-testid="stDataFrame"] {

    border-radius: 14px;

    overflow: hidden;

    border: 1px solid #e2e8f0;
}


/* ---------------------------------------------------------
   EXPANDER
--------------------------------------------------------- */

div[data-testid="stExpander"] {

    background: white;

    border-radius: 14px;

    border: 1px solid #e2e8f0;

    box-shadow:
        0 3px 10px rgba(15,23,42,0.04);
}


/* ---------------------------------------------------------
   SIDEBAR
--------------------------------------------------------- */

section[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #ecfeff 0%,
            #f0fdfa 50%,
            #eef2ff 100%
        );

    border-right:
        1px solid #dbeafe;
}

section[data-testid="stSidebar"] h1 {

    color: #0f766e;
}


/* ---------------------------------------------------------
   BUTTONS
--------------------------------------------------------- */

.stButton > button {

    border-radius: 10px;

    border: none;

    background:
        linear-gradient(
            90deg,
            #0f766e,
            #0891b2
        );

    color: white;

    font-weight: 600;

    transition: 0.2s ease;
}

.stButton > button:hover {

    transform: translateY(-2px);

    color: white;

    box-shadow:
        0 8px 18px rgba(8,145,178,0.20);
}


/* ---------------------------------------------------------
   DIVIDERS
--------------------------------------------------------- */

hr {

    margin-top: 2rem;
    margin-bottom: 2rem;

    border-color: #e2e8f0;
}


/* ---------------------------------------------------------
   MOBILE
--------------------------------------------------------- */

@media (max-width: 768px) {

    .pharma-hero {
        padding: 28px;
    }

    .pharma-hero-title {
        font-size: 36px;
    }

    .pharma-hero-subtitle {
        font-size: 18px;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================
#
# IMPORTANT:
# This HTML is intentionally kept on compact lines.
# Do not indent individual HTML lines inside the string.
# ============================================================

st.markdown(
    '<div class="pharma-hero">'
    '<div class="pharma-hero-title">💊 PharmaSignal AI</div>'
    '<div class="pharma-hero-subtitle">'
    'Explainable Drug-Safety Reporting Pattern Monitor'
    '</div>'
    '<div class="pharma-hero-text">'
    'Explore FDA Adverse Event Reporting System (FAERS) data '
    'through an interactive and easy-to-understand dashboard. '
    'PharmaSignal AI organizes large-scale drug-safety reporting '
    'data so users can explore medicines, adverse events, '
    'reporting frequency, and potential safety-signal candidates.'
    '</div>'
    '<div class="pharma-status">'
    '● &nbsp; PharmaSignal AI analytics ready'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("💊 PharmaSignal AI")

st.sidebar.caption(
    "Drug-Safety Analytics Dashboard"
)

st.sidebar.divider()


period = st.sidebar.selectbox(
    "Select FAERS period",
    [
        "2025",
        "2026 Q1",
    ],
)


st.sidebar.success(
    f"Currently exploring: {period}"
)


st.sidebar.markdown("### 🧭 Dashboard")

st.sidebar.markdown(
    """
**🏠 FAERS Explorer**

Explore drugs, adverse events and reporting patterns.

**📊 Method Comparison**

Compare machine-learning models with ROR, PRR and IC.

**🎯 Priority Matrix**

Explore and prioritize potential reporting patterns.
"""
)


st.sidebar.divider()


st.sidebar.warning(
    "FAERS reports describe suspected associations. "
    "They do not establish that a drug caused an event."
)


# ============================================================
# LOAD SUMMARY
# ============================================================

summary = get_faers_summary(period)


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.header("📊 FAERS Dataset Overview")

st.caption(
    f"Key statistics from the cleaned PharmaSignal AI "
    f"{period} FAERS dataset."
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "📄 Safety Reports",
        f"{summary['reports']:,}",
        help=(
            "Number of unique FAERS safety reports "
            "represented in this dataset."
        ),
    )


with col2:

    st.metric(
        "💊 Suspect Drugs",
        f"{summary['drugs']:,}",
        help=(
            "Number of unique suspect-drug names "
            "in the dataset."
        ),
    )


with col3:

    st.metric(
        "🩺 Adverse Events",
        f"{summary['events']:,}",
        help=(
            "Number of unique adverse-event terms "
            "in the dataset."
        ),
    )


with col4:

    st.metric(
        "🔗 Drug–Event Pairs",
        f"{summary['pairs']:,}",
        help=(
            "Number of unique suspect-drug and "
            "adverse-event combinations."
        ),
    )


st.info(
    "💡 **What am I looking at?**  \n"
    "A FAERS safety report may mention one or more medicines "
    "and one or more adverse events. PharmaSignal AI organizes "
    "these reports into drug–event combinations so reporting "
    "patterns are easier to explore."
)


# ============================================================
# TOP REPORTED PAIRS
# ============================================================

st.divider()

st.header(
    "🔥 Most Frequently Reported Drug–Event Pairs"
)

st.write(
    "Explore the drug–adverse-event combinations appearing "
    "in the largest number of unique FAERS reports."
)


top_n = st.slider(
    "Number of pairs to display",
    min_value=5,
    max_value=30,
    value=15,
    step=5,
)


with st.spinner(
    "Loading reported drug–event pairs..."
):

    top_pairs = get_top_pairs(
        period=period,
        n=top_n,
    )


if top_pairs.empty:

    st.warning(
        "No drug–event pairs were found."
    )

else:

    top_pairs = top_pairs.copy()


    top_pairs["pair"] = (

        top_pairs["suspect_drug"].astype(str)

        + " — "

        + top_pairs["adverse_event"].astype(str)

    )


    chart_data = (

        top_pairs

        .sort_values(
            "report_count",
            ascending=True,
        )

    )


    fig = px.bar(

        chart_data,

        x="report_count",

        y="pair",

        orientation="h",

        color="report_count",

        color_continuous_scale=[
            "#99f6e4",
            "#06b6d4",
            "#4f46e5",
        ],

        labels={
            "report_count":
                "Unique FAERS Reports",

            "pair":
                "Drug–Adverse Event",
        },

        title=(
            f"Top {top_n} Reported "
            f"Drug–Event Pairs — {period}"
        ),
    )


    fig.update_layout(

        height=620,

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        coloraxis_showscale=False,

        font=dict(
            color="#334155"
        ),

        title=dict(
            font=dict(
                size=20,
                color="#0f172a",
            )
        ),

        margin=dict(
            l=20,
            r=20,
            t=70,
            b=20,
        ),

        yaxis_title="",

        xaxis_title=(
            "Number of Unique FAERS Reports"
        ),
    )


    fig.update_xaxes(

        showgrid=True,

        gridcolor="rgba(148,163,184,0.18)",

        zeroline=False,
    )


    fig.update_yaxes(
        showgrid=False,
    )


    fig.update_traces(

        hovertemplate=(

            "<b>%{y}</b><br>"

            "Unique Reports: %{x:,}"

            "<extra></extra>"

        )
    )


    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    st.caption(
        "Higher reporting frequency means the combination "
        "appeared in more FAERS reports. It does not mean "
        "that the medicine caused the adverse event."
    )


    with st.expander(
        "📋 View Top Pairs Table"
    ):

        table = top_pairs[
            [
                "suspect_drug",
                "adverse_event",
                "report_count",
            ]
        ].copy()


        table.columns = [
            "Suspect Drug",
            "Adverse Event",
            "Unique Reports",
        ]


        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# INTERACTIVE EXPLORER
# ============================================================

st.divider()

st.header("🔎 Interactive FAERS Explorer")

st.write(
    "Choose a drug to explore its reported adverse events, "
    "or choose an adverse event to see the suspect drugs "
    "reported with it."
)


drug_tab, event_tab = st.tabs(
    [
        "💊 Explore by Drug",
        "🩺 Explore by Adverse Event",
    ]
)


# ============================================================
# DRUG TAB
# ============================================================

with drug_tab:

    st.subheader(
        "💊 Search by Suspect Drug"
    )


    st.caption(
        "Start typing inside the box to search "
        "available FAERS drug names."
    )


    with st.spinner(
        "Loading drug names..."
    ):

        drug_names = get_drug_names(
            period
        )


    selected_drug = st.selectbox(

        "Choose a drug",

        options=[
            "Select a drug..."
        ] + drug_names,

        key="drug_selector",
    )


    if selected_drug != "Select a drug...":

        with st.spinner(
            f"Exploring {selected_drug}..."
        ):

            drug_results = search_drug(
                period,
                selected_drug,
            )


        if drug_results.empty:

            st.warning(
                "No reporting information "
                "was found for this drug."
            )

        else:

            number_events = (
                drug_results[
                    "adverse_event"
                ]
                .nunique()
            )


            total_pair_reports = int(
                drug_results[
                    "report_count"
                ]
                .sum()
            )


            d1, d2 = st.columns(2)


            with d1:

                st.metric(
                    "🩺 Associated Event Terms",
                    f"{number_events:,}",
                )


            with d2:

                st.metric(
                    "📄 Pair-Level Report Count",
                    f"{total_pair_reports:,}",
                    help=(
                        "Sum of pair-level report counts. "
                        "This is not necessarily the number "
                        "of unique patients."
                    ),
                )


            st.subheader(
                f"Most Reported Events with {selected_drug}"
            )


            drug_chart_data = (

                drug_results

                .head(20)

                .sort_values(
                    "report_count",
                    ascending=True,
                )

            )


            fig_drug = px.bar(

                drug_chart_data,

                x="report_count",

                y="adverse_event",

                orientation="h",

                color="report_count",

                color_continuous_scale=[
                    "#bae6fd",
                    "#0ea5e9",
                    "#1d4ed8",
                ],

                labels={
                    "report_count":
                        "Unique Reports",

                    "adverse_event":
                        "Adverse Event",
                },
            )


            fig_drug.update_layout(

                height=600,

                paper_bgcolor="rgba(0,0,0,0)",

                plot_bgcolor="rgba(0,0,0,0)",

                coloraxis_showscale=False,

                yaxis_title="",

                xaxis_title=(
                    "Number of Unique Reports"
                ),

                margin=dict(
                    l=20,
                    r=20,
                    t=30,
                    b=20,
                ),
            )


            fig_drug.update_xaxes(

                showgrid=True,

                gridcolor=(
                    "rgba(148,163,184,0.18)"
                ),
            )


            fig_drug.update_yaxes(
                showgrid=False,
            )


            fig_drug.update_traces(

                hovertemplate=(

                    "<b>%{y}</b><br>"

                    "Reports: %{x:,}"

                    "<extra></extra>"

                )
            )


            st.plotly_chart(
                fig_drug,
                use_container_width=True,
            )


            with st.expander(
                "📋 View Detailed Drug Results"
            ):

                display_drug = (
                    drug_results.copy()
                )


                display_drug = (
                    display_drug.rename(
                        columns={
                            "adverse_event":
                                "Adverse Event",

                            "report_count":
                                "Unique Reports",

                            "ror":
                                "ROR",

                            "prr":
                                "PRR",

                            "ic":
                                "IC",

                            "meets_minimum_support":
                                "Meets Minimum Support",
                        }
                    )
                )


                st.dataframe(
                    display_drug,
                    use_container_width=True,
                    hide_index=True,
                )


# ============================================================
# EVENT TAB
# ============================================================

with event_tab:

    st.subheader(
        "🩺 Search by Adverse Event"
    )


    st.caption(
        "Start typing inside the box to search "
        "available FAERS adverse-event terms."
    )


    with st.spinner(
        "Loading adverse-event names..."
    ):

        event_names = get_event_names(
            period
        )


    selected_event = st.selectbox(

        "Choose an adverse event",

        options=[
            "Select an adverse event..."
        ] + event_names,

        key="event_selector",
    )


    if (
        selected_event
        !=
        "Select an adverse event..."
    ):

        with st.spinner(
            f"Exploring {selected_event}..."
        ):

            event_results = search_event(
                period,
                selected_event,
            )


        if event_results.empty:

            st.warning(
                "No reporting information "
                "was found for this adverse event."
            )

        else:

            number_drugs = (
                event_results[
                    "suspect_drug"
                ]
                .nunique()
            )


            total_pair_reports = int(
                event_results[
                    "report_count"
                ]
                .sum()
            )


            e1, e2 = st.columns(2)


            with e1:

                st.metric(
                    "💊 Associated Suspect Drugs",
                    f"{number_drugs:,}",
                )


            with e2:

                st.metric(
                    "📄 Pair-Level Report Count",
                    f"{total_pair_reports:,}",
                    help=(
                        "Sum of pair-level report counts. "
                        "This is not necessarily the number "
                        "of unique patients."
                    ),
                )


            st.subheader(
                f"Most Reported Drugs with "
                f"{selected_event}"
            )


            event_chart_data = (

                event_results

                .head(20)

                .sort_values(
                    "report_count",
                    ascending=True,
                )

            )


            fig_event = px.bar(

                event_chart_data,

                x="report_count",

                y="suspect_drug",

                orientation="h",

                color="report_count",

                color_continuous_scale=[
                    "#ddd6fe",
                    "#8b5cf6",
                    "#4f46e5",
                ],

                labels={
                    "report_count":
                        "Unique Reports",

                    "suspect_drug":
                        "Suspect Drug",
                },
            )


            fig_event.update_layout(

                height=600,

                paper_bgcolor="rgba(0,0,0,0)",

                plot_bgcolor="rgba(0,0,0,0)",

                coloraxis_showscale=False,

                yaxis_title="",

                xaxis_title=(
                    "Number of Unique Reports"
                ),

                margin=dict(
                    l=20,
                    r=20,
                    t=30,
                    b=20,
                ),
            )


            fig_event.update_xaxes(

                showgrid=True,

                gridcolor=(
                    "rgba(148,163,184,0.18)"
                ),
            )


            fig_event.update_yaxes(
                showgrid=False,
            )


            fig_event.update_traces(

                hovertemplate=(

                    "<b>%{y}</b><br>"

                    "Reports: %{x:,}"

                    "<extra></extra>"

                )
            )


            st.plotly_chart(
                fig_event,
                use_container_width=True,
            )


            with st.expander(
                "📋 View Detailed Event Results"
            ):

                display_event = (
                    event_results.copy()
                )


                display_event = (
                    display_event.rename(
                        columns={
                            "suspect_drug":
                                "Suspect Drug",

                            "report_count":
                                "Unique Reports",

                            "ror":
                                "ROR",

                            "prr":
                                "PRR",

                            "ic":
                                "IC",

                            "meets_minimum_support":
                                "Meets Minimum Support",
                        }
                    )
                )


                st.dataframe(
                    display_event,
                    use_container_width=True,
                    hide_index=True,
                )


# ============================================================
# HOW TO USE
# ============================================================

st.divider()

st.header("🧭 How to Use PharmaSignal AI")

st.write(
    "The dashboard follows a simple three-stage workflow."
)


step1, step2, step3 = st.columns(3)


with step1:

    st.subheader("🔎 1. Explore")

    st.write(
        "Explore drugs, adverse events and reporting "
        "frequency in the FAERS data."
    )

    st.info(
        "Use this page to understand what is present "
        "in the reporting data."
    )


with step2:

    st.subheader("📊 2. Compare")

    st.write(
        "Compare Logistic Regression, Random Forest "
        "and Gradient Boosting with ROR, PRR and IC."
    )

    st.info(
        "The Method Comparison page evaluates how well "
        "the different approaches prioritize the "
        "external reference controls."
    )


with step3:

    st.subheader("🎯 3. Prioritize")

    st.write(
        "Explore the Priority Matrix and ranking "
        "results to identify patterns for further review."
    )

    st.info(
        "Priority means review priority — "
        "not confirmed drug risk."
    )


# ============================================================
# INTERPRETATION NOTICE
# ============================================================

st.divider()

st.warning(
    "⚠️ **Important Scientific Interpretation**\n\n"
    "FAERS is a spontaneous-reporting system. Reporting "
    "frequency, ROR, PRR, IC, machine-learning predictions, "
    "and PharmaSignal AI rankings represent potential "
    "drug-safety reporting patterns. They do **not** establish "
    "that a medicine caused an adverse event.\n\n"
    "PharmaSignal AI is designed to support pharmacovigilance "
    "review and prioritization, not replace clinical, "
    "scientific, or regulatory judgment."
)


# ============================================================
# FOOTER
# ============================================================

st.divider()


footer1, footer2 = st.columns(
    [4, 1]
)


with footer1:

    st.caption(
        "💊 PharmaSignal AI | "
        "FAERS Drug-Safety Reporting Analytics | "
        "Research Dashboard"
    )


with footer2:

    st.caption(
        f"Dataset: {period}"
    )