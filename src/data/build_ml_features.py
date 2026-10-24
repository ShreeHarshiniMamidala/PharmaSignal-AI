from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# PharmaAI - Build ML Features at OMOP Control Level
# ============================================================
#
# PURPOSE
# -------
# Build one machine-learning row per matched OMOP drug-outcome
# control by aggregating evidence across its approved mapped
# FAERS adverse-event terms.
#
# WHY CONTROL LEVEL?
# ------------------
# One OMOP outcome may map to several FAERS MedDRA terms.
# Therefore, the mapped-reference file can contain multiple
# drug-event rows for the same OMOP drug-outcome control.
#
# Treating those rows as independent ML samples could create
# leakage and over-count one reference control.
#
# This script therefore creates:
#
#     ONE ROW = ONE OMOP DRUG-OUTCOME CONTROL
#
# LABEL
# -----
# groundTruth:
#     1 = OMOP positive control
#     0 = OMOP negative control
#
# IMPORTANT
# ---------
# ROR, PRR, IC, their thresholds, and related signal flags are
# NOT used as ML predictor features in this dataset.
#
# They are summarized separately so they can later be compared
# against the ML models on the same reference controls.
#
# No source signal table is modified.
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\mamid\OneDrive\Desktop\PharmaAI"
)

REFERENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reference"
)

ML_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "ml"
)

ML_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. INPUT FILES
# ============================================================

INPUT_FILES = {
    "2025": (
        REFERENCE_DIR
        / "pharmaai_mapped_reference_labels_2025.parquet"
    ),

    "2026_Q1": (
        REFERENCE_DIR
        / "pharmaai_mapped_reference_labels_2026_Q1.parquet"
    ),
}


# ============================================================
# 3. FEATURES
# ============================================================
#
# Predictor features are based on underlying FAERS support and
# prevalence counts, NOT ROR/PRR/IC.
#
# The mapped file should contain these columns from the existing
# signal table:
#
# pair_report_count
# drug_report_count
# event_report_count
# total_reports_in_signal_universe
# expected_pair_count
#
# expected_pair_count is retained as a count-based expectation,
# not a disproportionality score.
#
# For outcomes with several approved MedDRA terms, pair-level
# and event-level evidence is aggregated across the mapped terms.
# ============================================================

BASE_COUNT_COLUMNS = [
    "pair_report_count",
    "drug_report_count",
    "event_report_count",
    "total_reports_in_signal_universe",
    "expected_pair_count",
]


# ============================================================
# 4. HELPERS
# ============================================================

def require_columns(
    df,
    columns,
    dataset_name
):

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"\n{dataset_name} is missing required "
            f"column(s):\n{missing}\n\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )


def safe_log1p(series):
    """
    log(1 + x), with invalid/negative values converted to NaN.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    numeric = numeric.where(
        numeric >= 0
    )

    return np.log1p(
        numeric
    )


# ============================================================
# 5. LOAD MAPPED REFERENCE DATA
# ============================================================

def load_dataset(
    dataset_name,
    file_path
):

    print("\n" + "=" * 80)
    print(
        f"LOADING MAPPED REFERENCE DATA: "
        f"{dataset_name}"
    )
    print("=" * 80)

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nInput file not found:\n"
            f"{file_path}\n\n"
            "Run the corrected mapped-reference "
            "label script first."
        )

    df = pd.read_parquet(
        file_path
    )

    required = [
        "reference_control_key",
        "omop_drug",
        "omop_outcome",
        "groundTruth",
        "suspect_drug",
        "adverse_event",
    ] + BASE_COUNT_COLUMNS

    require_columns(
        df,
        required,
        dataset_name
    )

    # Convert numeric columns explicitly.
    for column in (
        BASE_COUNT_COLUMNS
        + ["groundTruth"]
    ):

        df[
            column
        ] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    if df[
        "groundTruth"
    ].isna().any():

        raise ValueError(
            f"{dataset_name}: groundTruth "
            "contains missing/non-numeric values."
        )

    invalid_labels = (
        ~df[
            "groundTruth"
        ].isin(
            [0, 1]
        )
    )

    if invalid_labels.any():

        raise ValueError(
            f"{dataset_name}: groundTruth must "
            "contain only 0 and 1."
        )

    print(
        f"\nMapped drug-event rows: "
        f"{len(df):,}"
    )

    print(
        "Unique OMOP controls: "
        f"{df['reference_control_key'].nunique():,}"
    )

    print(
        "Positive controls represented: "
        f"{df.loc[df['groundTruth'] == 1, 'reference_control_key'].nunique():,}"
    )

    print(
        "Negative controls represented: "
        f"{df.loc[df['groundTruth'] == 0, 'reference_control_key'].nunique():,}"
    )

    return df


# ============================================================
# 6. VALIDATE LABEL CONSISTENCY
# ============================================================

def validate_control_labels(
    df,
    dataset_name
):

    label_counts = (
        df
        .groupby(
            "reference_control_key"
        )[
            "groundTruth"
        ]
        .nunique()
    )

    conflicts = (
        label_counts[
            label_counts > 1
        ]
    )

    if not conflicts.empty:

        raise ValueError(
            f"\n{dataset_name}: "
            f"{len(conflicts):,} OMOP controls "
            "have conflicting groundTruth labels."
        )

    print(
        "\nGround-truth consistency check: PASS"
    )


# ============================================================
# 7. BUILD CONTROL-LEVEL FEATURES
# ============================================================

def build_control_features(
    df,
    dataset_name
):

    print("\n" + "=" * 80)
    print(
        f"BUILDING CONTROL-LEVEL FEATURES: "
        f"{dataset_name}"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Identity / label table
    # --------------------------------------------------------

    identity_columns = [
        "reference_control_key",
        "omop_drug",
        "omop_outcome",
        "groundTruth",
    ]

    if (
        "omop_outcome_source"
        in df.columns
    ):

        identity_columns.insert(
            3,
            "omop_outcome_source"
        )

    identity = (
        df[
            identity_columns
        ]
        .drop_duplicates(
            subset=[
                "reference_control_key"
            ]
        )
        .copy()
    )

    # --------------------------------------------------------
    # Number of approved mapped events observed for this
    # drug-outcome control.
    # --------------------------------------------------------

    mapped_event_features = (
        df
        .groupby(
            "reference_control_key",
            as_index=False
        )
        .agg(
            mapped_event_count=(
                "adverse_event",
                "nunique"
            ),
            mapped_pair_row_count=(
                "adverse_event",
                "size"
            ),
        )
    )

    # --------------------------------------------------------
    # Pair support features
    #
    # Several mapped events may belong to one OMOP outcome.
    # We summarize their observed pair support.
    # --------------------------------------------------------

    pair_features = (
        df
        .groupby(
            "reference_control_key",
            as_index=False
        )
        .agg(
            pair_count_sum=(
                "pair_report_count",
                "sum"
            ),
            pair_count_max=(
                "pair_report_count",
                "max"
            ),
            pair_count_mean=(
                "pair_report_count",
                "mean"
            ),
            pair_count_median=(
                "pair_report_count",
                "median"
            ),
            pair_count_min=(
                "pair_report_count",
                "min"
            ),
        )
    )

    # --------------------------------------------------------
    # Drug support
    #
    # drug_report_count should generally be the same for a drug
    # across mapped events in one signal universe. We use MAX
    # rather than SUM so the same drug support is not multiplied
    # simply because an outcome has multiple mapped terms.
    # --------------------------------------------------------

    drug_features = (
        df
        .groupby(
            "reference_control_key",
            as_index=False
        )
        .agg(
            drug_report_count=(
                "drug_report_count",
                "max"
            )
        )
    )

    # --------------------------------------------------------
    # Event prevalence
    #
    # Event counts differ across mapped MedDRA terms.
    # Sum/max/mean capture the overall outcome-event footprint.
    # --------------------------------------------------------

    event_features = (
        df
        .groupby(
            "reference_control_key",
            as_index=False
        )
        .agg(
            event_report_count_sum=(
                "event_report_count",
                "sum"
            ),
            event_report_count_max=(
                "event_report_count",
                "max"
            ),
            event_report_count_mean=(
                "event_report_count",
                "mean"
            ),
        )
    )

    # --------------------------------------------------------
    # Expected support
    #
    # This is the expected pair count already present in the
    # signal table. We aggregate it but do not use ROR/PRR/IC.
    # --------------------------------------------------------

    expected_features = (
        df
        .groupby(
            "reference_control_key",
            as_index=False
        )
        .agg(
            expected_pair_count_sum=(
                "expected_pair_count",
                "sum"
            ),
            expected_pair_count_max=(
                "expected_pair_count",
                "max"
            ),
            expected_pair_count_mean=(
                "expected_pair_count",
                "mean"
            ),
        )
    )

    # --------------------------------------------------------
    # Signal-universe size
    # --------------------------------------------------------

    universe_features = (
        df
        .groupby(
            "reference_control_key",
            as_index=False
        )
        .agg(
            total_reports=(
                "total_reports_in_signal_universe",
                "max"
            )
        )
    )

    # --------------------------------------------------------
    # Merge features
    # --------------------------------------------------------

    features = identity.copy()

    for table in [
        mapped_event_features,
        pair_features,
        drug_features,
        event_features,
        expected_features,
        universe_features,
    ]:

        features = features.merge(
            table,
            on="reference_control_key",
            how="left",
            validate="one_to_one",
        )

    # --------------------------------------------------------
    # Derived frequency features
    #
    # These are based on raw support/prevalence counts.
    # They are NOT ROR/PRR/IC.
    # --------------------------------------------------------

    features[
        "pair_support_rate"
    ] = (
        features[
            "pair_count_sum"
        ]
        / features[
            "total_reports"
        ]
    )

    features[
        "drug_reporting_rate"
    ] = (
        features[
            "drug_report_count"
        ]
        / features[
            "total_reports"
        ]
    )

    features[
        "event_reporting_rate"
    ] = (
        features[
            "event_report_count_sum"
        ]
        / features[
            "total_reports"
        ]
    )

    # --------------------------------------------------------
    # Log-transformed versions of highly skewed count features.
    # Keep both raw and log versions for later modeling choices.
    # --------------------------------------------------------

    log_sources = [
        "pair_count_sum",
        "pair_count_max",
        "pair_count_mean",
        "drug_report_count",
        "event_report_count_sum",
        "event_report_count_max",
        "expected_pair_count_sum",
        "expected_pair_count_max",
    ]

    for column in log_sources:

        features[
            f"log1p_{column}"
        ] = safe_log1p(
            features[
                column
            ]
        )

    features[
        "dataset"
    ] = dataset_name

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    if (
        features[
            "reference_control_key"
        ].duplicated().any()
    ):

        raise ValueError(
            f"{dataset_name}: duplicate "
            "reference controls remain after "
            "feature aggregation."
        )

    print(
        f"\nFinal ML rows: "
        f"{len(features):,}"
    )

    print(
        "Positive labels: "
        f"{int((features['groundTruth'] == 1).sum()):,}"
    )

    print(
        "Negative labels: "
        f"{int((features['groundTruth'] == 0).sum()):,}"
    )

    return features


# ============================================================
# 8. BUILD TRADITIONAL-METHOD COMPARISON TABLE
# ============================================================

def build_traditional_comparison(
    df,
    dataset_name
):
    """
    Keep ROR/PRR/IC separate from the ML predictor table.

    For a control with multiple mapped FAERS events, this table
    summarizes the traditional scores across those event terms.
    These values are for later baseline comparison, NOT for ML
    training in the independent experiment.
    """

    print("\n" + "=" * 80)
    print(
        f"BUILDING TRADITIONAL BASELINE TABLE: "
        f"{dataset_name}"
    )
    print("=" * 80)

    traditional_columns = [
        "ror",
        "prr",
        "ic",
    ]

    require_columns(
        df,
        traditional_columns,
        dataset_name
    )

    traditional = (
        df
        .groupby(
            "reference_control_key",
            as_index=False
        )
        .agg(
            ror_max=(
                "ror",
                "max"
            ),
            ror_mean=(
                "ror",
                "mean"
            ),
            prr_max=(
                "prr",
                "max"
            ),
            prr_mean=(
                "prr",
                "mean"
            ),
            ic_max=(
                "ic",
                "max"
            ),
            ic_mean=(
                "ic",
                "mean"
            ),
        )
    )

    labels = (
        df[
            [
                "reference_control_key",
                "omop_drug",
                "omop_outcome",
                "groundTruth",
            ]
        ]
        .drop_duplicates(
            subset=[
                "reference_control_key"
            ]
        )
    )

    traditional = labels.merge(
        traditional,
        on="reference_control_key",
        how="left",
        validate="one_to_one",
    )

    traditional[
        "dataset"
    ] = dataset_name

    print(
        f"\nTraditional baseline rows: "
        f"{len(traditional):,}"
    )

    return traditional


# ============================================================
# 9. SAVE OUTPUTS
# ============================================================

def save_outputs(
    dataset_name,
    features,
    traditional
):

    feature_csv = (
        ML_DIR
        / f"ml_features_{dataset_name}.csv"
    )

    feature_parquet = (
        ML_DIR
        / f"ml_features_{dataset_name}.parquet"
    )

    baseline_csv = (
        ML_DIR
        / (
            "traditional_baselines_"
            f"{dataset_name}.csv"
        )
    )

    baseline_parquet = (
        ML_DIR
        / (
            "traditional_baselines_"
            f"{dataset_name}.parquet"
        )
    )

    features.to_csv(
        feature_csv,
        index=False
    )

    features.to_parquet(
        feature_parquet,
        index=False
    )

    traditional.to_csv(
        baseline_csv,
        index=False
    )

    traditional.to_parquet(
        baseline_parquet,
        index=False
    )

    print(
        "\nSaved ML feature CSV:\n"
        f"{feature_csv}"
    )

    print(
        "\nSaved ML feature parquet:\n"
        f"{feature_parquet}"
    )

    print(
        "\nSaved traditional baseline CSV:\n"
        f"{baseline_csv}"
    )

    print(
        "\nSaved traditional baseline parquet:\n"
        f"{baseline_parquet}"
    )


# ============================================================
# 10. VALIDATION SUMMARY
# ============================================================

def make_summary(
    dataset_name,
    source_df,
    features
):

    predictor_columns = [
        column
        for column in features.columns
        if column not in {
            "reference_control_key",
            "omop_drug",
            "omop_outcome",
            "omop_outcome_source",
            "groundTruth",
            "dataset",
        }
    ]

    missing_predictor_cells = int(
        features[
            predictor_columns
        ].isna().sum().sum()
    )

    return {
        "dataset": dataset_name,
        "mapped_source_rows": len(
            source_df
        ),
        "unique_controls": len(
            features
        ),
        "positive_controls": int(
            (
                features[
                    "groundTruth"
                ] == 1
            ).sum()
        ),
        "negative_controls": int(
            (
                features[
                    "groundTruth"
                ] == 0
            ).sum()
        ),
        "predictor_feature_count": len(
            predictor_columns
        ),
        "missing_predictor_cells": (
            missing_predictor_cells
        ),
        "ror_used_as_ml_feature": False,
        "prr_used_as_ml_feature": False,
        "ic_used_as_ml_feature": False,
    }


# ============================================================
# 11. MAIN
# ============================================================

def main():

    print("\n" + "=" * 80)
    print("PHARMAAI - BUILD ML FEATURES")
    print("=" * 80)

    print(
        "\nDesign:"
    )

    print(
        "  One row = one matched OMOP "
        "drug-outcome control"
    )

    print(
        "  Label = OMOP groundTruth"
    )

    print(
        "  ROR/PRR/IC = excluded from "
        "independent ML predictors"
    )

    summaries = []

    for (
        dataset_name,
        file_path
    ) in INPUT_FILES.items():

        df = load_dataset(
            dataset_name,
            file_path
        )

        validate_control_labels(
            df,
            dataset_name
        )

        features = (
            build_control_features(
                df,
                dataset_name
            )
        )

        traditional = (
            build_traditional_comparison(
                df,
                dataset_name
            )
        )

        save_outputs(
            dataset_name,
            features,
            traditional
        )

        summaries.append(
            make_summary(
                dataset_name,
                df,
                features
            )
        )

    summary_df = pd.DataFrame(
        summaries
    )

    summary_file = (
        RESULTS_DIR
        / "ml_feature_build_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    print("\n" + "=" * 80)
    print("FINAL FEATURE SUMMARY")
    print("=" * 80)

    print(
        "\n"
        + summary_df.to_string(
            index=False
        )
    )

    print(
        "\n\nSummary saved:\n"
        f"{summary_file}"
    )

    print("\n" + "=" * 80)
    print("ML FEATURE BUILD COMPLETE")
    print("=" * 80)

    print(
        "\nNEXT STEP:"
    )

    print(
        "Inspect the feature summary and "
        "class counts before training models."
    )

    print(
        "Do not perform a random split of "
        "mapped event rows; this dataset is "
        "already aggregated to one row per "
        "OMOP reference control."
    )


# ============================================================
# 12. RUN
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\nFeature build interrupted."
        )

        sys.exit(1)

    except Exception as error:

        print(
            "\nFeature build failed:"
        )

        print(error)

        raise
