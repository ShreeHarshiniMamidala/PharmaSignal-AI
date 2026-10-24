from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PHARMAAI DASHBOARD DATA LOADER
# ============================================================
#
# IMPORTANT:
#
# The Streamlit dashboard does NOT load the huge canonical
# FAERS files:
#
#   faers_2025_clean.parquet
#   faers_2026_Q1_clean.parquet
#
# Those files contain millions of rows and make the dashboard
# unnecessarily slow.
#
# Instead, the dashboard uses the already-created FINAL
# pair-level traditional signal tables.
#
# These contain one row per drug-event pair and already include:
#
#   suspect_drug
#   adverse_event
#   pair_report_count
#   ROR
#   PRR
#   IC
#   support information
#
# This makes the dashboard much faster.
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)


# ============================================================
# FINAL PAIR-LEVEL SIGNAL TABLES
# ============================================================

TRADITIONAL_FILES = {

    "2025":
        PROCESSED_DIR
        / "traditional_full"
        / "full_traditional_signals_2025.parquet",

    "2026 Q1":
        PROCESSED_DIR
        / "traditional_full"
        / "full_traditional_signals_2026_Q1.parquet",
}


# ============================================================
# VALIDATED DATASET STATISTICS
# ============================================================
#
# These numbers come from the completed PharmaAI data
# validation / full traditional signal-building pipeline.
#
# Keeping these small summary values here prevents Streamlit
# from scanning 16+ million rows simply to calculate four
# numbers for the Home page.
# ============================================================

DATASET_SUMMARY = {

    "2025": {

        "reports": 1_390_248,

        "drugs": 11_188,

        "events": 16_541,

        "pairs": 1_476_886,

        "observations": 16_296_363,
    },

    "2026 Q1": {

        "reports": 369_049,

        "drugs": 7_456,

        "events": 12_085,

        "pairs": 702_328,

        "observations": 4_420_077,
    },
}


# ============================================================
# FILE CHECK
# ============================================================

def check_file(path):

    if not path.exists():

        st.error(
            f"""
            Required PharmaAI file was not found:

            {path}
            """
        )

        st.stop()


# ============================================================
# LOAD PAIR-LEVEL SIGNAL DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_traditional_signals(period):

    """
    Load the FINAL full-data pair-level signal table.

    This is much smaller than the canonical FAERS dataset.

    It is used for:

    - Home page exploration
    - Drug search
    - Event search
    - Top reported pairs
    - Priority Matrix
    - ROR / PRR / IC exploration
    """

    path = TRADITIONAL_FILES[period]

    check_file(path)

    columns = [

        "suspect_drug",

        "adverse_event",

        "pair_report_count",

        "ror",

        "prr",

        "ic",

        "meets_minimum_support",
    ]


    try:

        df = pd.read_parquet(
            path,
            columns=columns
        )

    except Exception:

        # Fallback in case one of the optional columns
        # has a slightly different name.

        df = pd.read_parquet(
            path
        )


    return df


# ============================================================
# DATASET SUMMARY
# ============================================================

def get_faers_summary(period):

    """
    Return validated PharmaAI dataset statistics.

    No large dataset is loaded.
    """

    summary = DATASET_SUMMARY[period]


    return {

        "reports":
            summary["reports"],

        "drugs":
            summary["drugs"],

        "events":
            summary["events"],

        "pairs":
            summary["pairs"],

        "rows":
            summary["observations"],
    }


# ============================================================
# TOP REPORTED DRUG-EVENT PAIRS
# ============================================================

@st.cache_data(show_spinner=False)
def get_top_pairs(period, n=20):

    """
    Return the most frequently reported drug-event pairs.

    IMPORTANT:

    We do NOT group the 16-million-row FAERS dataset.

    pair_report_count has already been calculated during
    the PharmaAI signal-building stage.
    """

    df = load_traditional_signals(
        period
    )


    required = {

        "suspect_drug",

        "adverse_event",

        "pair_report_count",
    }


    if not required.issubset(df.columns):

        missing = (
            required
            - set(df.columns)
        )

        st.error(
            "Signal table is missing required columns: "
            + ", ".join(missing)
        )

        st.stop()


    top_pairs = (

        df[
            [
                "suspect_drug",
                "adverse_event",
                "pair_report_count",
            ]
        ]

        .sort_values(
            "pair_report_count",
            ascending=False
        )

        .head(n)

        .rename(
            columns={
                "pair_report_count":
                    "report_count"
            }
        )

        .reset_index(
            drop=True
        )
    )


    return top_pairs


# ============================================================
# SEARCH BY DRUG
# ============================================================

@st.cache_data(show_spinner=False)
def search_drug(
    period,
    drug_name
):

    """
    Find adverse events reported with a selected drug.

    Searches the pair-level table rather than the
    multi-million-row canonical dataset.
    """

    df = load_traditional_signals(
        period
    )


    drug_name = (
        str(drug_name)
        .strip()
        .upper()
    )


    filtered = df[

        df[
            "suspect_drug"
        ]

        .astype(str)

        .str.strip()

        .str.upper()

        .eq(
            drug_name
        )

    ].copy()


    if filtered.empty:

        return pd.DataFrame()


    columns = [

        "adverse_event",

        "pair_report_count",
    ]


    optional_columns = [

        "ror",

        "prr",

        "ic",

        "meets_minimum_support",
    ]


    for column in optional_columns:

        if column in filtered.columns:

            columns.append(
                column
            )


    results = (

        filtered[
            columns
        ]

        .rename(
            columns={
                "pair_report_count":
                    "report_count"
            }
        )

        .sort_values(
            "report_count",
            ascending=False
        )

        .reset_index(
            drop=True
        )
    )


    return results


# ============================================================
# SEARCH BY ADVERSE EVENT
# ============================================================

@st.cache_data(show_spinner=False)
def search_event(
    period,
    event_name
):

    """
    Find suspect drugs reported with an adverse event.

    Searches the pair-level signal table.
    """

    df = load_traditional_signals(
        period
    )


    event_name = (
        str(event_name)
        .strip()
        .upper()
    )


    filtered = df[

        df[
            "adverse_event"
        ]

        .astype(str)

        .str.strip()

        .str.upper()

        .eq(
            event_name
        )

    ].copy()


    if filtered.empty:

        return pd.DataFrame()


    columns = [

        "suspect_drug",

        "pair_report_count",
    ]


    optional_columns = [

        "ror",

        "prr",

        "ic",

        "meets_minimum_support",
    ]


    for column in optional_columns:

        if column in filtered.columns:

            columns.append(
                column
            )


    results = (

        filtered[
            columns
        ]

        .rename(
            columns={
                "pair_report_count":
                    "report_count"
            }
        )

        .sort_values(
            "report_count",
            ascending=False
        )

        .reset_index(
            drop=True
        )
    )


    return results


# ============================================================
# AVAILABLE DRUG NAMES
# ============================================================

@st.cache_data(show_spinner=False)
def get_drug_names(period):

    """
    Return available suspect-drug names.

    Useful later for autocomplete/dropdown search.
    """

    df = load_traditional_signals(
        period
    )


    drugs = (

        df[
            "suspect_drug"
        ]

        .dropna()

        .astype(str)

        .drop_duplicates()

        .sort_values()

        .tolist()
    )


    return drugs


# ============================================================
# AVAILABLE ADVERSE EVENTS
# ============================================================

@st.cache_data(show_spinner=False)
def get_event_names(period):

    """
    Return available adverse-event names.

    Useful later for autocomplete/dropdown search.
    """

    df = load_traditional_signals(
        period
    )


    events = (

        df[
            "adverse_event"
        ]

        .dropna()

        .astype(str)

        .drop_duplicates()

        .sort_values()

        .tolist()
    )


    return events


# ============================================================
# PRIORITY MATRIX DATA
# ============================================================

@st.cache_data(show_spinner=False)
def get_priority_data(
    period,
    minimum_reports=5
):

    """
    Return data for the Priority Matrix.

    Only pairs meeting the selected report-support
    threshold are returned.
    """

    df = load_traditional_signals(
        period
    )


    filtered = df[

        df[
            "pair_report_count"
        ]

        >= minimum_reports

    ].copy()


    return filtered


# ============================================================
# SIGNAL DETAILS
# ============================================================

@st.cache_data(show_spinner=False)
def get_signal_details(
    period,
    drug_name,
    event_name
):

    """
    Retrieve one specific drug-event pair.

    This will later allow the dashboard to show:

    - report count
    - ROR
    - PRR
    - IC
    - support information
    """

    df = load_traditional_signals(
        period
    )


    drug_name = (
        str(drug_name)
        .strip()
        .upper()
    )


    event_name = (
        str(event_name)
        .strip()
        .upper()
    )


    result = df[

        (
            df[
                "suspect_drug"
            ]

            .astype(str)

            .str.strip()

            .str.upper()

            .eq(
                drug_name
            )
        )

        &

        (
            df[
                "adverse_event"
            ]

            .astype(str)

            .str.strip()

            .str.upper()

            .eq(
                event_name
            )
        )

    ].copy()


    return result