from pathlib import Path
import pandas as pd


# ============================================================
# PharmaAI - Diagnose Top-K Control Universe
# ============================================================
#
# PURPOSE
# -------
# Determine why the ML prediction files and traditional
# control-score files contain the same number of OMOP controls
# but different reference_control_key values.
#
# This script DOES NOT modify any existing data or results.
# It only creates diagnostics.
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\mamid\OneDrive\Desktop\PharmaAI"
)

RESULTS_ROOT = PROJECT_ROOT / "results"

ML_FILES = {
    "Logistic Regression": {
        "2025": (
            RESULTS_ROOT / "ml" / "logistic_regression"
            / "logistic_regression_predictions_2025_oof.csv"
        ),
        "2026_Q1": (
            RESULTS_ROOT / "ml" / "logistic_regression"
            / "logistic_regression_predictions_2026_Q1.csv"
        ),
    },
    "Random Forest": {
        "2025": (
            RESULTS_ROOT / "ml" / "random_forest"
            / "random_forest_predictions_2025_oof.csv"
        ),
        "2026_Q1": (
            RESULTS_ROOT / "ml" / "random_forest"
            / "random_forest_predictions_2026_Q1.csv"
        ),
    },
    "Gradient Boosting": {
        "2025": (
            RESULTS_ROOT / "ml" / "gradient_boosting"
            / "gradient_boosting_predictions_2025_oof.csv"
        ),
        "2026_Q1": (
            RESULTS_ROOT / "ml" / "gradient_boosting"
            / "gradient_boosting_predictions_2026_Q1.csv"
        ),
    },
}

TRADITIONAL_FILES = {
    "2025": (
        RESULTS_ROOT / "traditional_full" / "evaluation"
        / "traditional_control_scores_2025.csv"
    ),
    "2026_Q1": (
        RESULTS_ROOT / "traditional_full" / "evaluation"
        / "traditional_control_scores_2026_Q1.csv"
    ),
}

OUTPUT_DIR = (
    RESULTS_ROOT
    / "top_k_ranking"
    / "diagnostics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def find_column(df, candidates, description):
    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

        if candidate.lower() in lower:
            return lower[candidate.lower()]

    raise ValueError(
        f"Could not find {description}.\n"
        f"Tried: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def normalize_key(series):
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
        .str.replace(r"\.0(?=__|$)", "", regex=True)
    )


def load_file(path, source_name):
    require_file(path)

    df = pd.read_csv(path)

    key_col = find_column(
        df,
        [
            "reference_control_key",
            "control_key",
        ],
        "reference-control key",
    )

    label_col = find_column(
        df,
        [
            "groundTruth",
            "ground_truth",
            "label",
            "y_true",
        ],
        "ground-truth column",
    )

    out = pd.DataFrame({
        "raw_key": df[key_col].astype(str),
        "normalized_key": normalize_key(
            df[key_col]
        ),
        "groundTruth": pd.to_numeric(
            df[label_col],
            errors="raise",
        ).astype(int),
    })

    # Carry identity columns if present.
    for col in [
        "exposureId",
        "exposureName",
        "outcomeId",
        "outcomeName",
        "standardized_outcome",
        "omop_drug",
        "omop_outcome",
    ]:
        if col in df.columns:
            out[col] = df[col]

    out["source"] = source_name

    return out


def compare_pair(
    left,
    right,
    left_name,
    right_name,
    dataset,
):
    raw_left = set(left["raw_key"])
    raw_right = set(right["raw_key"])

    norm_left = set(left["normalized_key"])
    norm_right = set(right["normalized_key"])

    raw_overlap = len(
        raw_left & raw_right
    )

    normalized_overlap = len(
        norm_left & norm_right
    )

    print(
        f"\n{left_name} vs {right_name}"
    )
    print(
        f"  Left controls:  {len(raw_left):,}"
    )
    print(
        f"  Right controls: {len(raw_right):,}"
    )
    print(
        f"  Exact raw-key overlap: "
        f"{raw_overlap:,}"
    )
    print(
        f"  Normalized-key overlap: "
        f"{normalized_overlap:,}"
    )

    if norm_left == norm_right:
        print(
            "  RESULT: SAME control universe after "
            "key normalization."
        )
        status = "SAME_AFTER_NORMALIZATION"
    else:
        print(
            "  RESULT: DIFFERENT control universe even "
            "after key normalization."
        )
        status = "DIFFERENT_CONTROLS"

    left_only = left[
        ~left["normalized_key"].isin(
            norm_right
        )
    ].copy()

    right_only = right[
        ~right["normalized_key"].isin(
            norm_left
        )
    ].copy()

    safe_left = (
        left_name.lower()
        .replace(" ", "_")
    )

    safe_right = (
        right_name.lower()
        .replace(" ", "_")
    )

    if len(left_only):
        left_only.to_csv(
            OUTPUT_DIR
            / (
                f"{dataset}_{safe_left}"
                f"_only_vs_{safe_right}.csv"
            ),
            index=False,
        )

    if len(right_only):
        right_only.to_csv(
            OUTPUT_DIR
            / (
                f"{dataset}_{safe_right}"
                f"_only_vs_{safe_left}.csv"
            ),
            index=False,
        )

    return {
        "dataset": dataset,
        "left_source": left_name,
        "right_source": right_name,
        "left_controls": len(raw_left),
        "right_controls": len(raw_right),
        "raw_key_overlap": raw_overlap,
        "normalized_key_overlap":
            normalized_overlap,
        "left_only_after_normalization":
            len(norm_left - norm_right),
        "right_only_after_normalization":
            len(norm_right - norm_left),
        "status": status,
    }


def compare_labels(
    left,
    right,
    left_name,
    right_name,
    dataset,
):
    merged = left[
        [
            "normalized_key",
            "groundTruth",
        ]
    ].merge(
        right[
            [
                "normalized_key",
                "groundTruth",
            ]
        ],
        on="normalized_key",
        how="inner",
        suffixes=("_left", "_right"),
    )

    disagreements = merged[
        merged["groundTruth_left"]
        != merged["groundTruth_right"]
    ]

    print(
        f"  Label disagreements on matched keys: "
        f"{len(disagreements):,}"
    )

    if len(disagreements):
        disagreements.to_csv(
            OUTPUT_DIR
            / (
                f"{dataset}_label_disagreements_"
                f"{left_name.lower().replace(' ', '_')}_"
                f"vs_{right_name.lower().replace(' ', '_')}.csv"
            ),
            index=False,
        )

    return len(disagreements)


def main():
    header(
        "PHARMAAI - DIAGNOSE TOP-K CONTROL UNIVERSE"
    )

    summary_rows = []

    for dataset in [
        "2025",
        "2026_Q1",
    ]:
        header(
            f"DATASET: {dataset}"
        )

        ml_frames = {}

        for method, files in ML_FILES.items():
            ml_frames[method] = load_file(
                files[dataset],
                method,
            )

            print(
                f"{method}: "
                f"{len(ml_frames[method]):,} rows"
            )

        traditional = load_file(
            TRADITIONAL_FILES[dataset],
            "Traditional",
        )

        print(
            f"Traditional: "
            f"{len(traditional):,} rows"
        )

        # First verify the three ML models truly share
        # the same controls.
        reference_ml = ml_frames[
            "Logistic Regression"
        ]

        for method in [
            "Random Forest",
            "Gradient Boosting",
        ]:
            result = compare_pair(
                reference_ml,
                ml_frames[method],
                "Logistic Regression",
                method,
                dataset,
            )

            result["label_disagreements"] = (
                compare_labels(
                    reference_ml,
                    ml_frames[method],
                    "Logistic Regression",
                    method,
                    dataset,
                )
            )

            summary_rows.append(
                result
            )

        # Compare each ML model with the traditional
        # OMOP control table.
        for method, frame in ml_frames.items():
            result = compare_pair(
                frame,
                traditional,
                method,
                "Traditional",
                dataset,
            )

            result["label_disagreements"] = (
                compare_labels(
                    frame,
                    traditional,
                    method,
                    "Traditional",
                    dataset,
                )
            )

            summary_rows.append(
                result
            )

        # Save raw key examples for visual inspection.
        key_examples = pd.concat(
            [
                reference_ml[
                    [
                        "raw_key",
                        "normalized_key",
                    ]
                ]
                .head(20)
                .assign(
                    source="Logistic Regression"
                ),
                traditional[
                    [
                        "raw_key",
                        "normalized_key",
                    ]
                ]
                .head(20)
                .assign(
                    source="Traditional"
                ),
            ],
            ignore_index=True,
        )

        key_examples.to_csv(
            OUTPUT_DIR
            / f"{dataset}_key_examples.csv",
            index=False,
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        OUTPUT_DIR
        / "control_universe_diagnostic_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    header(
        "CONTROL UNIVERSE DIAGNOSTIC COMPLETE"
    )

    print(
        "\nSummary:"
    )

    print(
        summary[
            [
                "dataset",
                "left_source",
                "right_source",
                "raw_key_overlap",
                "normalized_key_overlap",
                "left_only_after_normalization",
                "right_only_after_normalization",
                "label_disagreements",
                "status",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nDiagnostic files saved to:\n"
        f"{OUTPUT_DIR}"
    )

    print(
        "\nINTERPRETATION:"
        "\n- SAME_AFTER_NORMALIZATION means the Top-K "
        "warning was only a key-format issue."
        "\n- DIFFERENT_CONTROLS means we must align all "
        "methods to their exact common OMOP controls before "
        "using the Top-K comparison."
    )


if __name__ == "__main__":
    main()
