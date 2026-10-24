from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PharmaAI - Visualize Existing Signal Tables
# ============================================================
#
# This script DOES NOT recalculate ROR, PRR, IC, or signal tables.
# It reads the already-created signal parquet files and creates
# descriptive visualizations only.
#
# Input:
#   data/processed/signals/drug_event_signals_2025.parquet
#   data/processed/signals/drug_event_signals_2026_Q1.parquet
#
# Output:
#   results/signals/2025/
#   results/signals/2026_Q1/
#
# IMPORTANT:
# Disproportionality patterns are potential signal candidates,
# not confirmed causal drug-adverse-event relationships.
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\mamid\OneDrive\Desktop\PharmaAI")

SIGNAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "signals"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "signals"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. INPUT FILES
# ============================================================

DATASETS = {
    "2025": (
        SIGNAL_DIR
        / "drug_event_signals_2025.parquet"
    ),

    "2026_Q1": (
        SIGNAL_DIR
        / "drug_event_signals_2026_Q1.parquet"
    ),
}

PROCESS_DATASETS = [
    "2025",
    "2026_Q1",
]

MIN_PAIR_REPORTS = 5
TOP_N = 20
DPI = 300


# ============================================================
# 3. HELPERS
# ============================================================

def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_subheader(title):
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


def save_figure(fig, output_file):
    fig.savefig(
        output_file,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output_file}")


# ============================================================
# 4. READ SIGNAL TABLE
# ============================================================

def read_signal_table(dataset_name, file_path):

    print_subheader(
        f"READING SIGNAL TABLE - {dataset_name}"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Signal table not found:\n{file_path}"
        )

    df = pd.read_parquet(file_path)

    required_columns = [
        "suspect_drug",
        "adverse_event",
        "pair_report_count",
        "ror",
        "ror_ci_lower",
        "ror_ci_upper",
        "prr",
        "ic",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    print(f"Rows loaded: {len(df):,}")

    return df


# ============================================================
# 5. PREPARE SUPPORTED PAIRS
# ============================================================

def prepare_supported_pairs(df):

    if "meets_minimum_support" in df.columns:

        supported = df[
            df["meets_minimum_support"] == True
        ].copy()

    else:

        supported = df[
            df["pair_report_count"]
            >= MIN_PAIR_REPORTS
        ].copy()

    print(
        f"Pairs with >= {MIN_PAIR_REPORTS} reports: "
        f"{len(supported):,}"
    )

    return supported


# ============================================================
# 6. VISUALIZATION 1
# Pair-report support distribution
# ============================================================

def plot_pair_support_distribution(
    supported,
    dataset_name,
    output_dir
):

    values = np.log10(
        supported[
            "pair_report_count"
        ].to_numpy(dtype=float)
        + 1
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.hist(
        values,
        bins=60
    )

    ax.set_title(
        "Drug-Event Pair Reporting Frequency "
        f"- {dataset_name}"
    )

    ax.set_xlabel(
        "log10(Pair Report Count + 1)"
    )

    ax.set_ylabel(
        "Number of Drug-Event Pairs"
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir
        / "01_pair_support_distribution.png"
    )


# ============================================================
# 7. VISUALIZATION 2
# ROR vs reporting support
# ============================================================

def plot_ror_vs_support(
    supported,
    dataset_name,
    output_dir
):

    plot_df = supported[
        (supported["ror"] > 0)
        & np.isfinite(
            supported["ror"]
        )
    ].copy()

    x = np.log10(
        plot_df[
            "pair_report_count"
        ].to_numpy(dtype=float)
        + 1
    )

    y = np.log10(
        plot_df[
            "ror"
        ].to_numpy(dtype=float)
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    hb = ax.hexbin(
        x,
        y,
        gridsize=70,
        mincnt=1,
        bins="log"
    )

    ax.axhline(
        y=0,
        linestyle="--",
        linewidth=1
    )

    ax.set_title(
        "Reporting Odds Ratio vs "
        "Reporting Support "
        f"- {dataset_name}"
    )

    ax.set_xlabel(
        "log10(Pair Report Count + 1)"
    )

    ax.set_ylabel(
        "log10(ROR)"
    )

    colorbar = fig.colorbar(
        hb,
        ax=ax
    )

    colorbar.set_label(
        "Drug-Event Pair Density"
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir
        / "02_ror_vs_support_hexbin.png"
    )


# ============================================================
# 8. VISUALIZATION 3
# ROR vs PRR
# ============================================================

def plot_ror_vs_prr(
    supported,
    dataset_name,
    output_dir
):

    plot_df = supported[
        (supported["ror"] > 0)
        & (supported["prr"] > 0)
        & np.isfinite(
            supported["ror"]
        )
        & np.isfinite(
            supported["prr"]
        )
    ].copy()

    x = np.log10(
        plot_df[
            "ror"
        ].to_numpy(dtype=float)
    )

    y = np.log10(
        plot_df[
            "prr"
        ].to_numpy(dtype=float)
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    hb = ax.hexbin(
        x,
        y,
        gridsize=70,
        mincnt=1,
        bins="log"
    )

    ax.axvline(
        x=0,
        linestyle="--",
        linewidth=1
    )

    ax.axhline(
        y=0,
        linestyle="--",
        linewidth=1
    )

    ax.set_title(
        "ROR vs PRR Across "
        "Drug-Event Pairs "
        f"- {dataset_name}"
    )

    ax.set_xlabel(
        "log10(ROR)"
    )

    ax.set_ylabel(
        "log10(PRR)"
    )

    colorbar = fig.colorbar(
        hb,
        ax=ax
    )

    colorbar.set_label(
        "Drug-Event Pair Density"
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir
        / "03_ror_vs_prr_hexbin.png"
    )


# ============================================================
# 9. VISUALIZATION 4
# IC vs reporting support
# ============================================================

def plot_ic_vs_support(
    supported,
    dataset_name,
    output_dir
):

    plot_df = supported[
        np.isfinite(
            supported["ic"]
        )
    ].copy()

    x = np.log10(
        plot_df[
            "pair_report_count"
        ].to_numpy(dtype=float)
        + 1
    )

    y = plot_df[
        "ic"
    ].to_numpy(dtype=float)

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    hb = ax.hexbin(
        x,
        y,
        gridsize=70,
        mincnt=1,
        bins="log"
    )

    ax.axhline(
        y=0,
        linestyle="--",
        linewidth=1
    )

    ax.set_title(
        "Information Component vs "
        "Reporting Support "
        f"- {dataset_name}"
    )

    ax.set_xlabel(
        "log10(Pair Report Count + 1)"
    )

    ax.set_ylabel(
        "Information Component (IC)"
    )

    colorbar = fig.colorbar(
        hb,
        ax=ax
    )

    colorbar.set_label(
        "Drug-Event Pair Density"
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir
        / "04_ic_vs_support_hexbin.png"
    )


# ============================================================
# 10. VISUALIZATION 5
# High-support disproportionate reporting candidates
# ============================================================

def plot_high_support_candidates(
    supported,
    dataset_name,
    output_dir
):

    candidates = supported[
        np.isfinite(
            supported["ror_ci_lower"]
        )
        & (
            supported[
                "ror_ci_lower"
            ]
            > 1
        )
    ].copy()

    top = (
        candidates
        .sort_values(
            [
                "pair_report_count",
                "ror_ci_lower"
            ],
            ascending=[
                False,
                False
            ]
        )
        .head(TOP_N)
        .copy()
    )

    if top.empty:

        print(
            "No eligible pairs for "
            "high-support candidate plot."
        )

        return

    top[
        "pair_name"
    ] = (
        top[
            "suspect_drug"
        ].astype(str)
        + " - "
        + top[
            "adverse_event"
        ].astype(str)
    )

    top = top.sort_values(
        "pair_report_count",
        ascending=True
    )

    fig, ax = plt.subplots(
        figsize=(12, 9)
    )

    ax.barh(
        top[
            "pair_name"
        ],
        top[
            "pair_report_count"
        ]
    )

    ax.set_title(
        "Frequently Reported "
        "Disproportionate Drug-Event Pairs "
        f"- {dataset_name}"
    )

    ax.set_xlabel(
        "Number of Reports"
    )

    ax.set_ylabel(
        "Drug-Event Pair"
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir
        / "05_high_support_disproportionate_pairs.png"
    )


# ============================================================
# 11. VISUALIZATION 6
# ROR confidence intervals
# ============================================================

def plot_ror_confidence_intervals(
    supported,
    dataset_name,
    output_dir
):

    candidates = supported[
        np.isfinite(
            supported["ror"]
        )
        & np.isfinite(
            supported["ror_ci_lower"]
        )
        & np.isfinite(
            supported["ror_ci_upper"]
        )
        & (
            supported[
                "ror_ci_lower"
            ]
            > 1
        )
    ].copy()

    top = (
        candidates
        .sort_values(
            [
                "pair_report_count",
                "ror_ci_lower"
            ],
            ascending=[
                False,
                False
            ]
        )
        .head(15)
        .copy()
    )

    if top.empty:

        print(
            "No eligible pairs for "
            "ROR confidence-interval plot."
        )

        return

    top[
        "pair_name"
    ] = (
        top[
            "suspect_drug"
        ].astype(str)
        + " - "
        + top[
            "adverse_event"
        ].astype(str)
    )

    top = top.sort_values(
        "ror",
        ascending=True
    )

    positions = np.arange(
        len(top)
    )

    lower_error = (
        top["ror"]
        - top["ror_ci_lower"]
    ).to_numpy(dtype=float)

    upper_error = (
        top["ror_ci_upper"]
        - top["ror"]
    ).to_numpy(dtype=float)

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    ax.errorbar(
        top[
            "ror"
        ].to_numpy(dtype=float),
        positions,
        xerr=np.vstack(
            [
                lower_error,
                upper_error
            ]
        ),
        fmt="o",
        capsize=3
    )

    ax.axvline(
        x=1,
        linestyle="--",
        linewidth=1
    )

    ax.set_yticks(
        positions
    )

    ax.set_yticklabels(
        top[
            "pair_name"
        ]
    )

    ax.set_xscale(
        "log"
    )

    ax.set_title(
        "ROR and 95% Confidence Intervals "
        f"- {dataset_name}"
    )

    ax.set_xlabel(
        "Reporting Odds Ratio (log scale)"
    )

    ax.set_ylabel(
        "Drug-Event Pair"
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir
        / "06_ror_confidence_intervals.png"
    )


# ============================================================
# 12. PROCESS DATASET
# ============================================================

def process_dataset(
    dataset_name,
    file_path
):

    print_header(
        f"PHARMAAI VISUALIZATIONS - "
        f"{dataset_name}"
    )

    df = read_signal_table(
        dataset_name,
        file_path
    )

    supported = prepare_supported_pairs(
        df
    )

    if supported.empty:
        print(
            "No supported pairs available. "
            "Skipping dataset."
        )
        return

    output_dir = (
        RESULTS_DIR
        / dataset_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    plot_pair_support_distribution(
        supported,
        dataset_name,
        output_dir
    )

    plot_ror_vs_support(
        supported,
        dataset_name,
        output_dir
    )

    plot_ror_vs_prr(
        supported,
        dataset_name,
        output_dir
    )

    plot_ic_vs_support(
        supported,
        dataset_name,
        output_dir
    )

    plot_high_support_candidates(
        supported,
        dataset_name,
        output_dir
    )

    plot_ror_confidence_intervals(
        supported,
        dataset_name,
        output_dir
    )

    print_subheader(
        "VISUALIZATION COMPLETE"
    )

    print(
        f"Output folder:\n"
        f"{output_dir}"
    )


# ============================================================
# 13. MAIN
# ============================================================

def main():

    print_header(
        "PharmaAI - Signal Visualization"
    )

    print(
        "\nThis script reads existing "
        "signal tables."
    )

    print(
        "ROR, PRR, and IC are NOT "
        "recalculated."
    )

    print(
        "\nOutput root:\n"
        f"{RESULTS_DIR}"
    )

    for dataset_name in PROCESS_DATASETS:

        if dataset_name not in DATASETS:

            print(
                f"Unknown dataset: "
                f"{dataset_name}"
            )

            continue

        process_dataset(
            dataset_name,
            DATASETS[
                dataset_name
            ]
        )

    print_header(
        "ALL PHARMAAI VISUALIZATIONS COMPLETE"
    )


# ============================================================
# 14. RUN
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\nVisualization interrupted "
            "by user."
        )

        sys.exit(1)

    except Exception as error:

        print(
            "\nVisualization failed:"
        )

        print(error)

        raise
