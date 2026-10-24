from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


# ============================================================
# PharmaAI - Evaluate Full Traditional Baselines
# ============================================================
#
# PURPOSE
# -------
# Evaluate the FULL-DATA ROR, PRR, and IC signal scores against
# the SAME OMOP reference controls used for the ML experiment.
#
# Inputs:
#   1. OMOP reference set
#   2. Approved OMOP -> FAERS outcome mapping
#   3. Full 2025 ROR/PRR/IC signal table
#   4. Full 2026 Q1 ROR/PRR/IC signal table
#
# IMPORTANT
# ---------
# - ROR / PRR / IC are NOT recalculated here.
# - OMOP groundTruth is the external reference label.
# - Unknown/unmatched FAERS pairs are NOT treated as negatives.
# - One final row is produced per matched OMOP control.
# - AUROC and PR-AUC are the primary ranking metrics.
# - No threshold is tuned on 2026 Q1.
#
# Aggregation rule:
#   An OMOP outcome can map to multiple approved FAERS events.
#   For each OMOP drug-outcome control, the MAXIMUM score among
#   its observed approved mapped FAERS events is used.
#
# This rule is fixed for ROR, PRR, and IC and for both periods.
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

RAW_REFERENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "reference"
)

TRADITIONAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "traditional_full"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "traditional_full"
    / "evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OMOP_FILE = (
    RAW_REFERENCE_DIR
    / "omop_reference_set.csv"
)

MAPPING_FILE = (
    REFERENCE_DIR
    / "omop_faers_outcome_mapping_approved.csv"
)

SIGNALS_2025 = (
    TRADITIONAL_DIR
    / "full_traditional_signals_2025.parquet"
)

SIGNALS_2026 = (
    TRADITIONAL_DIR
    / "full_traditional_signals_2026_Q1.parquet"
)


# Explicit source OMOP outcome -> standardized project outcome.
OUTCOME_TRANSLATION = {
    "OMOP ACUTE LIVER FAILURE 1":
        "Acute Liver Injury",
    "OMOP ACUTE RENAL FAILURE 1":
        "Acute Kidney Injury",
    "OMOP ACUTE MYOCARDIAL INFARCTION 1":
        "Acute Myocardial Infarction",
    "HOI UPPER GI #3":
        "GI Bleed",
}


def header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def normalize_text_value(value):
    if pd.isna(value):
        return pd.NA

    value = str(value).strip().upper()

    # Collapse repeated whitespace so the known OMOP AMI
    # double-space variation does not cause a mismatch.
    value = " ".join(value.split())

    return value


def normalize_series(series):
    return series.map(
        normalize_text_value
    )


def detect_column(
    columns,
    candidates,
    label,
):
    lower_lookup = {
        str(column).lower(): column
        for column in columns
    }

    for candidate in candidates:
        if candidate in columns:
            return candidate

        if candidate.lower() in lower_lookup:
            return lower_lookup[
                candidate.lower()
            ]

    raise ValueError(
        f"Could not identify {label} column.\n"
        f"Tried: {candidates}\n"
        f"Available columns: {list(columns)}"
    )


def load_omop():
    header("LOADING OMOP REFERENCE SET")

    require_file(OMOP_FILE)

    omop = pd.read_csv(
        OMOP_FILE
    )

    exposure_id_col = detect_column(
        omop.columns,
        ["exposureId", "exposure_id"],
        "OMOP exposure ID",
    )

    exposure_name_col = detect_column(
        omop.columns,
        ["exposureName", "exposure_name"],
        "OMOP exposure name",
    )

    outcome_id_col = detect_column(
        omop.columns,
        ["outcomeId", "outcome_id"],
        "OMOP outcome ID",
    )

    outcome_name_col = detect_column(
        omop.columns,
        ["outcomeName", "outcome_name"],
        "OMOP outcome name",
    )

    ground_truth_col = detect_column(
        omop.columns,
        [
            "groundTruth",
            "ground_truth",
        ],
        "ground truth",
    )

    omop = omop.rename(
        columns={
            exposure_id_col: "exposureId",
            exposure_name_col: "exposureName",
            outcome_id_col: "outcomeId",
            outcome_name_col: "outcomeName",
            ground_truth_col: "groundTruth",
        }
    )

    omop["groundTruth"] = pd.to_numeric(
        omop["groundTruth"],
        errors="raise",
    ).astype(int)

    unexpected_labels = sorted(
        set(omop["groundTruth"])
        - {0, 1}
    )

    if unexpected_labels:
        raise ValueError(
            "Unexpected OMOP labels: "
            f"{unexpected_labels}"
        )

    omop[
        "normalized_exposure"
    ] = normalize_series(
        omop["exposureName"]
    )

    omop[
        "normalized_source_outcome"
    ] = normalize_series(
        omop["outcomeName"]
    )

    translation_normalized = {
        normalize_text_value(key): value
        for key, value
        in OUTCOME_TRANSLATION.items()
    }

    omop[
        "standardized_outcome"
    ] = omop[
        "normalized_source_outcome"
    ].map(
        translation_normalized
    )

    unmapped_outcomes = (
        omop.loc[
            omop[
                "standardized_outcome"
            ].isna(),
            "outcomeName",
        ]
        .drop_duplicates()
        .tolist()
    )

    if unmapped_outcomes:
        raise ValueError(
            "Unexpected OMOP outcomes without explicit "
            "translation:\n"
            + "\n".join(
                f"  - {x}"
                for x in unmapped_outcomes
            )
        )

    omop[
        "reference_control_key"
    ] = (
        omop["exposureId"].astype(str)
        + "__"
        + omop["outcomeId"].astype(str)
    )

    duplicate_controls = omop[
        "reference_control_key"
    ].duplicated().sum()

    if duplicate_controls:
        raise ValueError(
            "Duplicate OMOP control keys found: "
            f"{duplicate_controls}"
        )

    print(
        f"\nOMOP controls: {len(omop):,}"
    )
    print(
        f"Positive controls: "
        f"{(omop['groundTruth'] == 1).sum():,}"
    )
    print(
        f"Negative controls: "
        f"{(omop['groundTruth'] == 0).sum():,}"
    )
    print(
        f"Unique drugs: "
        f"{omop['exposureName'].nunique():,}"
    )
    print(
        f"Unique outcomes: "
        f"{omop['outcomeName'].nunique():,}"
    )

    return omop


def load_mapping():
    header("LOADING APPROVED OUTCOME MAPPING")

    require_file(MAPPING_FILE)

    mapping = pd.read_csv(
        MAPPING_FILE
    )

    outcome_col = detect_column(
        mapping.columns,
        [
            "omop_outcome",
            "standardized_outcome",
        ],
        "standardized OMOP outcome",
    )

    event_col = detect_column(
        mapping.columns,
        [
            "faers_adverse_event",
            "adverse_event",
        ],
        "FAERS adverse event",
    )

    mapping = mapping.rename(
        columns={
            outcome_col: "standardized_outcome",
            event_col: "faers_adverse_event",
        }
    )

    if "mapping_status" in mapping.columns:
        approved = mapping[
            mapping["mapping_status"]
            .astype(str)
            .str.upper()
            .str.startswith("APPROVED")
        ].copy()

        if len(approved) == 0:
            raise ValueError(
                "Mapping file contains mapping_status, "
                "but no APPROVED rows were found."
            )

        mapping = approved

    mapping[
        "normalized_faers_event"
    ] = normalize_series(
        mapping["faers_adverse_event"]
    )

    mapping[
        "standardized_outcome"
    ] = (
        mapping[
            "standardized_outcome"
        ]
        .astype(str)
        .str.strip()
    )

    mapping = mapping[
        [
            "standardized_outcome",
            "faers_adverse_event",
            "normalized_faers_event",
        ]
    ].drop_duplicates()

    print(
        f"\nApproved mapping rows: "
        f"{len(mapping):,}"
    )

    print("\nApproved terms per outcome:")
    print(
        mapping.groupby(
            "standardized_outcome"
        ).size().to_string()
    )

    return mapping


def expand_reference_controls(
    omop,
    mapping,
):
    header("EXPANDING OMOP CONTROLS THROUGH APPROVED MAPPING")

    expanded = omop.merge(
        mapping,
        on="standardized_outcome",
        how="left",
        validate="many_to_many",
    )

    missing_mapping = expanded[
        "normalized_faers_event"
    ].isna()

    if missing_mapping.any():
        missing = (
            expanded.loc[
                missing_mapping,
                "standardized_outcome",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Some standardized outcomes have no approved "
            f"FAERS mapping: {missing}"
        )

    print(
        f"\nOMOP controls: "
        f"{omop['reference_control_key'].nunique():,}"
    )

    print(
        f"Expanded control-event rows: "
        f"{len(expanded):,}"
    )

    return expanded


def load_signal_table(
    path,
    dataset_name,
):
    header(
        f"LOADING FULL TRADITIONAL SIGNALS: {dataset_name}"
    )

    require_file(path)

    needed = [
        "suspect_drug",
        "adverse_event",
        "pair_report_count",
        "ror",
        "prr",
        "ic",
    ]

    signals = pd.read_parquet(
        path,
        columns=needed,
    )

    signals[
        "normalized_drug"
    ] = normalize_series(
        signals["suspect_drug"]
    )

    signals[
        "normalized_event"
    ] = normalize_series(
        signals["adverse_event"]
    )

    for metric in [
        "ror",
        "prr",
        "ic",
    ]:
        signals[metric] = pd.to_numeric(
            signals[metric],
            errors="coerce",
        )

        signals[metric] = signals[
            metric
        ].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    print(
        f"\nSignal rows: {len(signals):,}"
    )

    print(
        f"Unique normalized drug-event pairs: "
        f"{signals[['normalized_drug', 'normalized_event']].drop_duplicates().shape[0]:,}"
    )

    print(
        f"Missing ROR: "
        f"{signals['ror'].isna().sum():,}"
    )
    print(
        f"Missing PRR: "
        f"{signals['prr'].isna().sum():,}"
    )
    print(
        f"Missing IC: "
        f"{signals['ic'].isna().sum():,}"
    )

    return signals


def match_controls(
    expanded,
    signals,
    dataset_name,
):
    header(
        f"MATCHING OMOP CONTROLS TO {dataset_name}"
    )

    reference = expanded.copy()

    reference[
        "normalized_drug"
    ] = reference[
        "normalized_exposure"
    ]

    matched = reference.merge(
        signals,
        left_on=[
            "normalized_drug",
            "normalized_faers_event",
        ],
        right_on=[
            "normalized_drug",
            "normalized_event",
        ],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    observed = matched[
        matched["_merge"] == "both"
    ].copy()

    observed_controls = observed[
        "reference_control_key"
    ].nunique()

    positive_controls = observed.loc[
        observed["groundTruth"] == 1,
        "reference_control_key",
    ].nunique()

    negative_controls = observed.loc[
        observed["groundTruth"] == 0,
        "reference_control_key",
    ].nunique()

    total_controls = expanded[
        "reference_control_key"
    ].nunique()

    print(
        f"\nMatched mapped drug-event rows: "
        f"{len(observed):,}"
    )

    print(
        f"Unique matched OMOP controls: "
        f"{observed_controls:,}/{total_controls:,}"
    )

    print(
        f"Matched positive controls: "
        f"{positive_controls:,}"
    )

    print(
        f"Matched negative controls: "
        f"{negative_controls:,}"
    )

    print(
        f"Control coverage: "
        f"{observed_controls / total_controls:.2%}"
    )

    return observed, matched


def aggregate_to_control(
    observed,
    dataset_name,
):
    header(
        f"AGGREGATING {dataset_name} TO ONE ROW PER OMOP CONTROL"
    )

    if observed.empty:
        raise ValueError(
            f"No OMOP controls matched for {dataset_name}."
        )

    identity_columns = [
        "reference_control_key",
        "exposureId",
        "exposureName",
        "outcomeId",
        "outcomeName",
        "standardized_outcome",
        "groundTruth",
    ]

    control = (
        observed.groupby(
            identity_columns,
            dropna=False,
            as_index=False,
        )
        .agg(
            mapped_event_rows=(
                "normalized_faers_event",
                "size",
            ),
            observed_mapped_event_count=(
                "normalized_faers_event",
                "nunique",
            ),
            pair_report_count_sum=(
                "pair_report_count",
                "sum",
            ),
            ror_score=(
                "ror",
                "max",
            ),
            prr_score=(
                "prr",
                "max",
            ),
            ic_score=(
                "ic",
                "max",
            ),
        )
    )

    duplicates = control[
        "reference_control_key"
    ].duplicated().sum()

    if duplicates:
        raise ValueError(
            "Aggregation did not produce one row per "
            f"OMOP control. Duplicates: {duplicates}"
        )

    print(
        f"\nFinal control rows: "
        f"{len(control):,}"
    )

    print(
        f"Positive controls: "
        f"{(control['groundTruth'] == 1).sum():,}"
    )

    print(
        f"Negative controls: "
        f"{(control['groundTruth'] == 0).sum():,}"
    )

    return control


def evaluate_metric(
    control_df,
    metric_column,
    method_name,
    dataset_name,
):
    valid = control_df[
        [
            "groundTruth",
            metric_column,
        ]
    ].dropna()

    n = len(valid)
    positives = int(
        (valid["groundTruth"] == 1).sum()
    )
    negatives = int(
        (valid["groundTruth"] == 0).sum()
    )

    if positives == 0 or negatives == 0:
        raise ValueError(
            f"{method_name} {dataset_name} evaluation "
            "does not contain both positive and negative "
            "controls."
        )

    y_true = valid[
        "groundTruth"
    ].astype(int)

    scores = valid[
        metric_column
    ].astype(float)

    auroc = roc_auc_score(
        y_true,
        scores,
    )

    pr_auc = average_precision_score(
        y_true,
        scores,
    )

    return {
        "dataset": dataset_name,
        "method": method_name,
        "aggregation": "maximum_observed_mapped_event_score",
        "n_controls": n,
        "positive_controls": positives,
        "negative_controls": negatives,
        "missing_score_controls": int(
            control_df[
                metric_column
            ].isna().sum()
        ),
        "auroc": auroc,
        "pr_auc": pr_auc,
    }


def evaluate_dataset(
    expanded,
    signal_path,
    dataset_name,
):
    signals = load_signal_table(
        signal_path,
        dataset_name,
    )

    observed, all_matches = match_controls(
        expanded,
        signals,
        dataset_name,
    )

    control_df = aggregate_to_control(
        observed,
        dataset_name,
    )

    header(
        f"EVALUATING ROR / PRR / IC: {dataset_name}"
    )

    results = []

    metric_specs = [
        ("ror_score", "ROR"),
        ("prr_score", "PRR"),
        ("ic_score", "IC"),
    ]

    for metric_column, method_name in metric_specs:
        result = evaluate_metric(
            control_df,
            metric_column,
            method_name,
            dataset_name,
        )

        results.append(result)

        print(
            f"\n{method_name}"
            f"\n  Controls: {result['n_controls']:,}"
            f"\n  Positive: {result['positive_controls']:,}"
            f"\n  Negative: {result['negative_controls']:,}"
            f"\n  Missing scores: {result['missing_score_controls']:,}"
            f"\n  AUROC:  {result['auroc']:.4f}"
            f"\n  PR-AUC: {result['pr_auc']:.4f}"
        )

    safe_name = dataset_name.replace(
        " ",
        "_",
    )

    control_path = (
        OUTPUT_DIR
        / f"traditional_control_scores_{safe_name}.csv"
    )

    control_df.to_csv(
        control_path,
        index=False,
    )

    unmatched_control_keys = set(
        expanded["reference_control_key"]
    ) - set(
        observed["reference_control_key"]
    )

    unmatched = (
        expanded[
            expanded[
                "reference_control_key"
            ].isin(
                unmatched_control_keys
            )
        ][
            [
                "reference_control_key",
                "exposureId",
                "exposureName",
                "outcomeId",
                "outcomeName",
                "standardized_outcome",
                "groundTruth",
            ]
        ]
        .drop_duplicates(
            subset=[
                "reference_control_key"
            ]
        )
    )

    unmatched_path = (
        OUTPUT_DIR
        / f"unmatched_omop_controls_{safe_name}.csv"
    )

    unmatched.to_csv(
        unmatched_path,
        index=False,
    )

    print(
        f"\nControl-level scores saved:\n"
        f"{control_path}"
    )

    print(
        f"\nUnmatched controls saved:\n"
        f"{unmatched_path}"
    )

    return results, control_df


def main():
    header(
        "PHARMAAI - EVALUATE FULL TRADITIONAL BASELINES"
    )

    print(
        "\nThis script evaluates the FULL-DATA "
        "ROR, PRR, and IC values."
    )

    print(
        "\nGround truth:"
        "\n  OMOP positive controls = 1"
        "\n  OMOP negative controls = 0"
    )

    print(
        "\nPrimary ranking metrics:"
        "\n  AUROC"
        "\n  PR-AUC"
    )

    print(
        "\nControl aggregation rule:"
        "\n  MAX score across observed approved mapped "
        "FAERS events for each OMOP control."
    )

    omop = load_omop()
    mapping = load_mapping()

    expanded = expand_reference_controls(
        omop,
        mapping,
    )

    all_results = []

    results_2025, _ = evaluate_dataset(
        expanded,
        SIGNALS_2025,
        "2025",
    )

    all_results.extend(
        results_2025
    )

    results_2026, _ = evaluate_dataset(
        expanded,
        SIGNALS_2026,
        "2026_Q1",
    )

    all_results.extend(
        results_2026
    )

    comparison = pd.DataFrame(
        all_results
    )

    comparison_path = (
        OUTPUT_DIR
        / "traditional_baseline_evaluation.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    header("TRADITIONAL BASELINE EVALUATION COMPLETE")

    print("\nFinal traditional baseline results:")
    print(
        comparison[
            [
                "dataset",
                "method",
                "n_controls",
                "positive_controls",
                "negative_controls",
                "auroc",
                "pr_auc",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        f"\nFinal evaluation table saved:\n"
        f"{comparison_path}"
    )

    print(
        "\nIMPORTANT:"
        "\nROR, PRR, and IC were evaluated against the "
        "same external OMOP ground truth used for ML."
    )

    print(
        "\nNEXT:"
        "\nCompare these traditional ranking results with "
        "Logistic Regression, Random Forest, and "
        "Gradient Boosting."
    )


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )
    main()
