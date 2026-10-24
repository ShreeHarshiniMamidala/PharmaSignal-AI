from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd


# ============================================================
# PharmaAI - Build Full Traditional Baselines
# ============================================================
#
# PURPOSE
# -------
# Rebuild ROR, PRR, and IC from the FULL canonical cleaned
# FAERS datasets:
#
#   2025 full year
#   2026 Q1
#
# This script does NOT use the old preliminary ROR evaluation.
# It creates a fresh full-data traditional signal table.
#
# IMPORTANT
# ---------
# This script assumes the canonical cleaned parquet files contain
# one row per report-suspect_drug-adverse_event relationship and
# contain:
#
#   safetyreportid
#   suspect_drug
#   adverse_event
#
# If your ID column has a slightly different name, the script
# attempts a few common alternatives.
#
# ROR / PRR / IC are calculated from UNIQUE REPORT-level
# drug-event occurrence, so duplicate occurrences inside the same
# report do not inflate the contingency table.
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\mamid\OneDrive\Desktop\PharmaAI"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_2025 = (
    PROCESSED_DIR / "faers_2025_clean.parquet"
)

INPUT_2026 = (
    PROCESSED_DIR / "faers_2026_Q1_clean.parquet"
)

OUTPUT_DIR = (
    PROCESSED_DIR / "traditional_full"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "traditional_full"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MIN_PAIR_REPORTS = 5


def header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found:\n{path}"
        )


def detect_column(columns, candidates, label):
    for candidate in candidates:
        if candidate in columns:
            return candidate

    lower_lookup = {
        str(column).lower(): column
        for column in columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]

    raise ValueError(
        f"Could not identify {label} column.\n"
        f"Tried: {candidates}\n"
        f"Available columns: {list(columns)}"
    )


def normalize_text(series):
    return (
        series
        .astype("string")
        .str.strip()
        .str.upper()
        .replace(
            {
                "": pd.NA,
                "NAN": pd.NA,
                "NONE": pd.NA,
                "<NA>": pd.NA,
            }
        )
    )


def load_canonical_dataset(path, dataset_name):
    header(f"LOADING {dataset_name}")

    require_file(path)

    # Read schema first so we only load the three columns needed
    # for the traditional signal calculation.
    try:
        import pyarrow.parquet as pq

        schema = pq.read_schema(path)
        available_columns = schema.names
    except Exception:
        available_columns = pd.read_parquet(
            path
        ).columns.tolist()

    report_col = detect_column(
        available_columns,
        [
            "safetyreportid",
            "safety_report_id",
            "report_id",
            "reportid",
        ],
        "report ID",
    )

    drug_col = detect_column(
        available_columns,
        [
            "suspect_drug",
            "drug",
            "drug_name",
        ],
        "suspect drug",
    )

    event_col = detect_column(
        available_columns,
        [
            "adverse_event",
            "reaction",
            "reaction_pt",
            "event",
        ],
        "adverse event",
    )

    print(f"\nReport column: {report_col}")
    print(f"Drug column:   {drug_col}")
    print(f"Event column:  {event_col}")

    df = pd.read_parquet(
        path,
        columns=[
            report_col,
            drug_col,
            event_col,
        ],
    )

    original_rows = len(df)

    print(
        f"\nCanonical rows loaded: "
        f"{original_rows:,}"
    )

    df = df.rename(
        columns={
            report_col: "report_id",
            drug_col: "suspect_drug",
            event_col: "adverse_event",
        }
    )

    df["report_id"] = (
        df["report_id"]
        .astype("string")
        .str.strip()
    )

    df["suspect_drug"] = normalize_text(
        df["suspect_drug"]
    )

    df["adverse_event"] = normalize_text(
        df["adverse_event"]
    )

    before_missing = len(df)

    df = df.dropna(
        subset=[
            "report_id",
            "suspect_drug",
            "adverse_event",
        ]
    )

    missing_removed = (
        before_missing - len(df)
    )

    # One report contributes at most once to a given
    # drug-event pair.
    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=[
            "report_id",
            "suspect_drug",
            "adverse_event",
        ]
    )

    duplicate_pair_rows_removed = (
        before_duplicates - len(df)
    )

    print(
        f"Rows removed for missing required values: "
        f"{missing_removed:,}"
    )

    print(
        "Duplicate report-drug-event rows removed: "
        f"{duplicate_pair_rows_removed:,}"
    )

    print(
        f"Unique report-drug-event rows used: "
        f"{len(df):,}"
    )

    print(
        f"Unique reports: "
        f"{df['report_id'].nunique():,}"
    )

    print(
        f"Unique suspect drugs: "
        f"{df['suspect_drug'].nunique():,}"
    )

    print(
        f"Unique adverse events: "
        f"{df['adverse_event'].nunique():,}"
    )

    return df, {
        "input_rows": original_rows,
        "missing_required_rows_removed": missing_removed,
        "duplicate_report_drug_event_rows_removed":
            duplicate_pair_rows_removed,
    }


def build_counts(df):
    header("BUILDING REPORT-LEVEL COUNTS")

    total_reports = int(
        df["report_id"].nunique()
    )

    print(
        f"\nTotal unique reports in signal universe: "
        f"{total_reports:,}"
    )

    pair_counts = (
        df.groupby(
            [
                "suspect_drug",
                "adverse_event",
            ],
            sort=False,
            observed=True,
        )["report_id"]
        .nunique()
        .rename("pair_report_count")
        .reset_index()
    )

    drug_counts = (
        df[
            [
                "report_id",
                "suspect_drug",
            ]
        ]
        .drop_duplicates()
        .groupby(
            "suspect_drug",
            sort=False,
            observed=True,
        )["report_id"]
        .nunique()
        .rename("drug_report_count")
        .reset_index()
    )

    event_counts = (
        df[
            [
                "report_id",
                "adverse_event",
            ]
        ]
        .drop_duplicates()
        .groupby(
            "adverse_event",
            sort=False,
            observed=True,
        )["report_id"]
        .nunique()
        .rename("event_report_count")
        .reset_index()
    )

    signals = pair_counts.merge(
        drug_counts,
        on="suspect_drug",
        how="left",
        validate="many_to_one",
    )

    signals = signals.merge(
        event_counts,
        on="adverse_event",
        how="left",
        validate="many_to_one",
    )

    signals[
        "total_reports_in_signal_universe"
    ] = total_reports

    print(
        f"Unique drug-event pairs: "
        f"{len(signals):,}"
    )

    return signals


def calculate_disproportionality(signals):
    header("CALCULATING ROR / PRR / IC")

    out = signals.copy()

    # 2x2 contingency table:
    #
    #                    Event       Other events
    # Drug                 a              b
    # Other drugs          c              d
    #
    # Counts are report-level counts.

    out["a"] = (
        out["pair_report_count"]
        .astype(float)
    )

    out["b"] = (
        out["drug_report_count"]
        - out["pair_report_count"]
    ).astype(float)

    out["c"] = (
        out["event_report_count"]
        - out["pair_report_count"]
    ).astype(float)

    out["d"] = (
        out["total_reports_in_signal_universe"]
        - out["a"]
        - out["b"]
        - out["c"]
    ).astype(float)

    invalid = (
        (out[["a", "b", "c", "d"]] < 0)
        .any(axis=1)
    )

    if invalid.any():
        example = out.loc[
            invalid,
            [
                "suspect_drug",
                "adverse_event",
                "a",
                "b",
                "c",
                "d",
            ],
        ].head()

        raise ValueError(
            "Invalid negative contingency counts detected.\n"
            f"{example}"
        )

    # --------------------------------------------------------
    # ROR
    # --------------------------------------------------------
    # Apply 0.5 Haldane-Anscombe correction only when any cell
    # of the 2x2 table is zero.
    zero_cell = (
        (out[["a", "b", "c", "d"]] == 0)
        .any(axis=1)
    )

    out[
        "continuity_correction_applied"
    ] = zero_cell

    a_ror = out["a"].copy()
    b_ror = out["b"].copy()
    c_ror = out["c"].copy()
    d_ror = out["d"].copy()

    a_ror.loc[zero_cell] += 0.5
    b_ror.loc[zero_cell] += 0.5
    c_ror.loc[zero_cell] += 0.5
    d_ror.loc[zero_cell] += 0.5

    out["ror"] = (
        (a_ror * d_ror)
        / (b_ror * c_ror)
    )

    log_ror = np.log(
        out["ror"]
    )

    ror_se = np.sqrt(
        (1.0 / a_ror)
        + (1.0 / b_ror)
        + (1.0 / c_ror)
        + (1.0 / d_ror)
    )

    out["ror_ci_lower"] = np.exp(
        log_ror - 1.96 * ror_se
    )

    out["ror_ci_upper"] = np.exp(
        log_ror + 1.96 * ror_se
    )

    # --------------------------------------------------------
    # PRR
    # --------------------------------------------------------
    drug_event_rate = (
        out["a"]
        / (out["a"] + out["b"])
    )

    other_drug_event_rate = (
        out["c"]
        / (out["c"] + out["d"])
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        out["prr"] = (
            drug_event_rate
            / other_drug_event_rate
        )

    # --------------------------------------------------------
    # Expected count and IC
    # --------------------------------------------------------
    out["expected_pair_count"] = (
        out["drug_report_count"]
        * out["event_report_count"]
        / out[
            "total_reports_in_signal_universe"
        ]
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        out["ic"] = np.log2(
            out["pair_report_count"]
            / out["expected_pair_count"]
        )

    # --------------------------------------------------------
    # Traditional support/signal flags
    # --------------------------------------------------------
    out["meets_minimum_support"] = (
        out["pair_report_count"]
        >= MIN_PAIR_REPORTS
    )

    out["ror_lower_ci_gt_1"] = (
        out["ror_ci_lower"] > 1.0
    )

    out["prr_gt_2"] = (
        out["prr"] > 2.0
    )

    # Replace non-finite values with NaN rather than silently
    # storing +/- infinity.
    for column in [
        "ror",
        "ror_ci_lower",
        "ror_ci_upper",
        "prr",
        "expected_pair_count",
        "ic",
    ]:
        out[column] = out[column].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    print(
        f"\nPairs meeting minimum support "
        f"(>= {MIN_PAIR_REPORTS} reports): "
        f"{out['meets_minimum_support'].sum():,}"
    )

    print(
        "Pairs with ROR lower 95% CI > 1: "
        f"{out['ror_lower_ci_gt_1'].sum():,}"
    )

    print(
        "Pairs with PRR > 2: "
        f"{out['prr_gt_2'].sum():,}"
    )

    print(
        "\nMissing/non-finite metric values:"
    )

    for column in [
        "ror",
        "prr",
        "ic",
    ]:
        print(
            f"  {column.upper()}: "
            f"{out[column].isna().sum():,}"
        )

    return out


def save_outputs(
    signals,
    dataset_name,
    load_stats,
):
    header(f"SAVING {dataset_name} OUTPUTS")

    safe_name = dataset_name.replace(
        " ",
        "_",
    )

    parquet_path = (
        OUTPUT_DIR
        / f"full_traditional_signals_{safe_name}.parquet"
    )

    csv_summary_path = (
        RESULTS_DIR
        / f"full_traditional_summary_{safe_name}.csv"
    )

    signals.to_parquet(
        parquet_path,
        index=False,
    )

    summary = {
        "dataset": dataset_name,
        **load_stats,
        "signal_pairs": int(len(signals)),
        "unique_suspect_drugs": int(
            signals["suspect_drug"].nunique()
        ),
        "unique_adverse_events": int(
            signals["adverse_event"].nunique()
        ),
        "total_reports": int(
            signals[
                "total_reports_in_signal_universe"
            ].iloc[0]
        ),
        "pairs_meeting_min_support": int(
            signals[
                "meets_minimum_support"
            ].sum()
        ),
        "pairs_ror_lower_ci_gt_1": int(
            signals[
                "ror_lower_ci_gt_1"
            ].sum()
        ),
        "pairs_prr_gt_2": int(
            signals[
                "prr_gt_2"
            ].sum()
        ),
        "missing_ror": int(
            signals["ror"].isna().sum()
        ),
        "missing_prr": int(
            signals["prr"].isna().sum()
        ),
        "missing_ic": int(
            signals["ic"].isna().sum()
        ),
    }

    pd.DataFrame(
        [summary]
    ).to_csv(
        csv_summary_path,
        index=False,
    )

    print(
        f"\nFull traditional signal table saved:\n"
        f"{parquet_path}"
    )

    print(
        f"\nSummary saved:\n"
        f"{csv_summary_path}"
    )

    return summary


def process_dataset(
    input_path,
    dataset_name,
):
    header(
        f"PROCESSING FULL DATASET: {dataset_name}"
    )

    df, load_stats = load_canonical_dataset(
        input_path,
        dataset_name,
    )

    signals = build_counts(df)

    signals = calculate_disproportionality(
        signals
    )

    summary = save_outputs(
        signals,
        dataset_name,
        load_stats,
    )

    # Free the very large row-level dataframe before the
    # next dataset is loaded.
    del df

    return summary


def main():
    header(
        "PHARMAAI - BUILD FULL TRADITIONAL BASELINES"
    )

    print(
        "\nThis run uses the FULL canonical datasets."
    )

    print(
        "\nInputs:"
        f"\n  2025:    {INPUT_2025}"
        f"\n  2026 Q1: {INPUT_2026}"
    )

    print(
        "\nMethods to calculate:"
        "\n  ROR"
        "\n  PRR"
        "\n  IC"
    )

    summaries = []

    summaries.append(
        process_dataset(
            INPUT_2025,
            "2025",
        )
    )

    summaries.append(
        process_dataset(
            INPUT_2026,
            "2026_Q1",
        )
    )

    combined_summary = pd.DataFrame(
        summaries
    )

    combined_summary_path = (
        RESULTS_DIR
        / "full_traditional_build_summary.csv"
    )

    combined_summary.to_csv(
        combined_summary_path,
        index=False,
    )

    header("FULL TRADITIONAL BASELINES COMPLETE")

    print(
        f"\nCombined summary saved:\n"
        f"{combined_summary_path}"
    )

    print(
        "\nIMPORTANT:"
        "\nThese files contain the full-data ROR, PRR, and IC "
        "values."
        "\nThey have NOT yet been evaluated against OMOP "
        "ground truth."
    )

    print(
        "\nNEXT:"
        "\nMap these full traditional signal tables to the "
        "same OMOP controls used for the ML models, then "
        "evaluate ROR, PRR, and IC."
    )


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )
    main()
