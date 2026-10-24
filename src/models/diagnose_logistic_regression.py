from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PharmaAI - Diagnose Logistic Regression
# ============================================================
#
# PURPOSE
# -------
# Diagnose why Logistic Regression performance changed from
# 2025 development evaluation to 2026 Q1 temporal evaluation.
#
# THIS SCRIPT DOES NOT:
#   - retrain the model
#   - tune hyperparameters
#   - change the 0.50 threshold
#   - modify ROR, PRR, or IC
#
# IT CHECKS:
#   1. 2025 vs 2026 feature distributions
#   2. standardized Logistic Regression coefficients
#   3. predicted-probability distributions by true class
#   4. 2026 false negatives
#   5. 2026 controls not represented in matched 2025 controls
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\mamid\OneDrive\Desktop\PharmaAI"
)

ML_DIR = PROJECT_ROOT / "data" / "processed" / "ml"

LR_RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "ml"
    / "logistic_regression"
)

DIAGNOSTIC_DIR = (
    LR_RESULTS_DIR
    / "diagnostics"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "logistic_regression"
    / "logistic_regression_2025.joblib"
)

TRAIN_FILE = ML_DIR / "ml_features_2025.parquet"
TEMPORAL_FILE = ML_DIR / "ml_features_2026_Q1.parquet"

PREDICTOR_FILE = (
    LR_RESULTS_DIR
    / "logistic_regression_predictors.json"
)

PREDICTION_2025_FILE = (
    LR_RESULTS_DIR
    / "logistic_regression_predictions_2025_oof.csv"
)

PREDICTION_2026_FILE = (
    LR_RESULTS_DIR
    / "logistic_regression_predictions_2026_Q1.csv"
)

COEFFICIENT_FILE = (
    LR_RESULTS_DIR
    / "logistic_regression_coefficients.csv"
)

DIAGNOSTIC_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TARGET = "groundTruth"
THRESHOLD = 0.50


def header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def load_inputs():
    header("LOADING LOGISTIC REGRESSION DIAGNOSTIC INPUTS")

    required_files = [
        TRAIN_FILE,
        TEMPORAL_FILE,
        MODEL_FILE,
        PREDICTOR_FILE,
        PREDICTION_2025_FILE,
        PREDICTION_2026_FILE,
        COEFFICIENT_FILE,
    ]

    for path in required_files:
        require_file(path)

    train_df = pd.read_parquet(TRAIN_FILE)
    temporal_df = pd.read_parquet(TEMPORAL_FILE)

    predictors = json.loads(
        PREDICTOR_FILE.read_text(
            encoding="utf-8"
        )
    )

    predictions_2025 = pd.read_csv(
        PREDICTION_2025_FILE
    )

    predictions_2026 = pd.read_csv(
        PREDICTION_2026_FILE
    )

    coefficients = pd.read_csv(
        COEFFICIENT_FILE
    )

    model = joblib.load(
        MODEL_FILE
    )

    print(f"\n2025 feature rows: {len(train_df):,}")
    print(f"2026 Q1 feature rows: {len(temporal_df):,}")
    print(f"Predictors: {len(predictors):,}")
    print(f"2025 prediction rows: {len(predictions_2025):,}")
    print(f"2026 prediction rows: {len(predictions_2026):,}")
    print(f"Coefficient rows: {len(coefficients):,}")
    print("Saved model loaded: YES")

    return (
        train_df,
        temporal_df,
        predictors,
        predictions_2025,
        predictions_2026,
        coefficients,
        model,
    )


def compare_feature_distributions(
    train_df,
    temporal_df,
    predictors,
):
    header("COMPARING 2025 VS 2026 Q1 FEATURE DISTRIBUTIONS")

    rows = []

    for feature in predictors:
        x_2025 = pd.to_numeric(
            train_df[feature],
            errors="coerce",
        )

        x_2026 = pd.to_numeric(
            temporal_df[feature],
            errors="coerce",
        )

        mean_2025 = x_2025.mean()
        mean_2026 = x_2026.mean()

        median_2025 = x_2025.median()
        median_2026 = x_2026.median()

        std_2025 = x_2025.std(ddof=1)
        std_2026 = x_2026.std(ddof=1)

        if (
            pd.notna(std_2025)
            and std_2025 > 0
        ):
            standardized_mean_shift = (
                mean_2026 - mean_2025
            ) / std_2025
        else:
            standardized_mean_shift = np.nan

        if (
            pd.notna(mean_2025)
            and mean_2025 != 0
        ):
            relative_mean_change = (
                mean_2026 - mean_2025
            ) / abs(mean_2025)
        else:
            relative_mean_change = np.nan

        rows.append(
            {
                "feature": feature,
                "mean_2025": mean_2025,
                "mean_2026_Q1": mean_2026,
                "median_2025": median_2025,
                "median_2026_Q1": median_2026,
                "std_2025": std_2025,
                "std_2026_Q1": std_2026,
                "standardized_mean_shift": (
                    standardized_mean_shift
                ),
                "absolute_standardized_mean_shift": (
                    abs(standardized_mean_shift)
                    if pd.notna(standardized_mean_shift)
                    else np.nan
                ),
                "relative_mean_change": relative_mean_change,
            }
        )

    comparison = pd.DataFrame(rows)

    comparison = comparison.sort_values(
        "absolute_standardized_mean_shift",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    output = (
        DIAGNOSTIC_DIR
        / "feature_distribution_shift.csv"
    )

    comparison.to_csv(
        output,
        index=False,
    )

    print(
        "\nLargest standardized mean shifts "
        "(2026 mean relative to 2025 SD):"
    )

    print(
        comparison[
            [
                "feature",
                "mean_2025",
                "mean_2026_Q1",
                "standardized_mean_shift",
            ]
        ]
        .head(10)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(f"\nSaved:\n{output}")

    return comparison


def inspect_coefficients(coefficients):
    header("INSPECTING LOGISTIC REGRESSION COEFFICIENTS")

    required = {
        "feature",
        "coefficient",
        "absolute_coefficient",
    }

    missing = required.difference(
        coefficients.columns
    )

    if missing:
        raise ValueError(
            "Coefficient file is missing columns: "
            f"{sorted(missing)}"
        )

    ordered = coefficients.sort_values(
        "absolute_coefficient",
        ascending=False,
    ).reset_index(drop=True)

    output = (
        DIAGNOSTIC_DIR
        / "coefficients_ranked.csv"
    )

    ordered.to_csv(
        output,
        index=False,
    )

    print(
        "\nTop 10 coefficients by absolute magnitude:"
    )

    print(
        ordered[
            [
                "feature",
                "coefficient",
                "absolute_coefficient",
            ]
        ]
        .head(10)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\nInterpretation reminder:"
        "\n  Positive coefficient -> pushes probability toward class 1"
        "\n  Negative coefficient -> pushes probability toward class 0"
        "\n  Coefficients are based on standardized predictors."
        "\n  Correlated features can make individual coefficients unstable."
    )

    print(f"\nSaved:\n{output}")

    return ordered


def probability_summary(
    predictions,
    dataset_name,
):
    required = {
        TARGET,
        "predicted_probability",
        "predicted_label",
    }

    missing = required.difference(
        predictions.columns
    )

    if missing:
        raise ValueError(
            f"{dataset_name} predictions are missing: "
            f"{sorted(missing)}"
        )

    rows = []

    for label in [0, 1]:
        subset = predictions[
            predictions[TARGET] == label
        ]["predicted_probability"]

        rows.append(
            {
                "dataset": dataset_name,
                "groundTruth": label,
                "n": len(subset),
                "probability_mean": subset.mean(),
                "probability_median": subset.median(),
                "probability_min": subset.min(),
                "probability_q25": subset.quantile(0.25),
                "probability_q75": subset.quantile(0.75),
                "probability_max": subset.max(),
                "fraction_predicted_positive": (
                    (subset >= THRESHOLD).mean()
                ),
            }
        )

    return rows


def analyze_probability_shift(
    predictions_2025,
    predictions_2026,
):
    header("ANALYZING PREDICTED-PROBABILITY DISTRIBUTIONS")

    rows = []

    rows.extend(
        probability_summary(
            predictions_2025,
            "2025_OOF",
        )
    )

    rows.extend(
        probability_summary(
            predictions_2026,
            "2026_Q1",
        )
    )

    summary = pd.DataFrame(rows)

    output = (
        DIAGNOSTIC_DIR
        / "probability_distribution_by_class.csv"
    )

    summary.to_csv(
        output,
        index=False,
    )

    print(
        "\n"
        + summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    positive_2025 = summary[
        (summary["dataset"] == "2025_OOF")
        & (summary["groundTruth"] == 1)
    ].iloc[0]

    positive_2026 = summary[
        (summary["dataset"] == "2026_Q1")
        & (summary["groundTruth"] == 1)
    ].iloc[0]

    print(
        "\nPositive-control probability shift:"
    )

    print(
        "  2025 positive mean probability: "
        f"{positive_2025['probability_mean']:.4f}"
    )

    print(
        "  2026 positive mean probability: "
        f"{positive_2026['probability_mean']:.4f}"
    )

    print(
        "  2025 positives predicted positive: "
        f"{positive_2025['fraction_predicted_positive']:.2%}"
    )

    print(
        "  2026 positives predicted positive: "
        f"{positive_2026['fraction_predicted_positive']:.2%}"
    )

    print(f"\nSaved:\n{output}")

    return summary


def analyze_2026_errors(predictions_2026):
    header("ANALYZING 2026 Q1 ERRORS")

    df = predictions_2026.copy()

    df["error_type"] = np.select(
        [
            (df[TARGET] == 1)
            & (df["predicted_label"] == 0),

            (df[TARGET] == 0)
            & (df["predicted_label"] == 1),

            (df[TARGET] == 1)
            & (df["predicted_label"] == 1),

            (df[TARGET] == 0)
            & (df["predicted_label"] == 0),
        ],
        [
            "False Negative",
            "False Positive",
            "True Positive",
            "True Negative",
        ],
        default="Unknown",
    )

    error_counts = (
        df["error_type"]
        .value_counts()
        .rename_axis("error_type")
        .reset_index(name="count")
    )

    print("\n2026 Q1 prediction outcomes:")
    print(
        error_counts.to_string(
            index=False
        )
    )

    false_negatives = df[
        df["error_type"] == "False Negative"
    ].copy()

    false_negatives = false_negatives.sort_values(
        "predicted_probability",
        ascending=True,
    )

    fn_output = (
        DIAGNOSTIC_DIR
        / "false_negatives_2026_Q1.csv"
    )

    false_negatives.to_csv(
        fn_output,
        index=False,
    )

    print(
        f"\nFalse negatives: "
        f"{len(false_negatives):,}"
    )

    if len(false_negatives) > 0:
        columns = [
            "omop_drug",
            "omop_outcome",
            "predicted_probability",
        ]

        if "seen_in_2025" in false_negatives.columns:
            columns.append(
                "seen_in_2025"
            )

        print(
            "\nLowest-probability false negatives:"
        )

        print(
            false_negatives[
                columns
            ]
            .head(10)
            .to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    print(f"\nSaved:\n{fn_output}")

    return df


def analyze_new_2026_controls(predictions_2026):
    header("ANALYZING 2026 Q1 NEW CONTROLS")

    if "seen_in_2025" not in predictions_2026.columns:
        print(
            "\nColumn 'seen_in_2025' is unavailable. "
            "Skipping new-control diagnostic."
        )
        return None

    seen = (
        predictions_2026[
            "seen_in_2025"
        ]
        .astype(str)
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
            }
        )
    )

    if seen.isna().any():
        # If pandas already loaded booleans, recover them directly.
        if pd.api.types.is_bool_dtype(
            predictions_2026["seen_in_2025"]
        ):
            seen = predictions_2026[
                "seen_in_2025"
            ]
        else:
            raise ValueError(
                "Could not interpret seen_in_2025 values."
            )

    new_controls = predictions_2026[
        ~seen
    ].copy()

    output = (
        DIAGNOSTIC_DIR
        / "new_controls_2026_Q1.csv"
    )

    new_controls.to_csv(
        output,
        index=False,
    )

    print(
        f"\nNew 2026 controls: "
        f"{len(new_controls):,}"
    )

    print(
        "Positive: "
        f"{(new_controls[TARGET] == 1).sum():,}"
    )

    print(
        "Negative: "
        f"{(new_controls[TARGET] == 0).sum():,}"
    )

    print(
        "Predicted positive: "
        f"{(new_controls['predicted_label'] == 1).sum():,}"
    )

    print(
        "Predicted negative: "
        f"{(new_controls['predicted_label'] == 0).sum():,}"
    )

    if len(new_controls) > 0:
        print(
            "\nNew-control predicted probabilities:"
        )

        print(
            new_controls[
                [
                    "omop_drug",
                    "omop_outcome",
                    TARGET,
                    "predicted_probability",
                    "predicted_label",
                ]
            ]
            .sort_values(
                "predicted_probability",
                ascending=False,
            )
            .to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    print(f"\nSaved:\n{output}")

    return new_controls


def create_combined_diagnostic(
    feature_shift,
    coefficients,
):
    header("COMBINING FEATURE SHIFT WITH MODEL COEFFICIENTS")

    combined = feature_shift.merge(
        coefficients[
            [
                "feature",
                "coefficient",
                "absolute_coefficient",
            ]
        ],
        on="feature",
        how="left",
        validate="one_to_one",
    )

    combined[
        "shift_x_coefficient"
    ] = (
        combined[
            "standardized_mean_shift"
        ]
        * combined[
            "coefficient"
        ]
    )

    combined[
        "absolute_shift_x_coefficient"
    ] = combined[
        "shift_x_coefficient"
    ].abs()

    combined = combined.sort_values(
        "absolute_shift_x_coefficient",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    output = (
        DIAGNOSTIC_DIR
        / "feature_shift_coefficient_diagnostic.csv"
    )

    combined.to_csv(
        output,
        index=False,
    )

    print(
        "\nFeatures with the largest combination of "
        "distribution shift and coefficient magnitude:"
    )

    print(
        combined[
            [
                "feature",
                "standardized_mean_shift",
                "coefficient",
                "shift_x_coefficient",
            ]
        ]
        .head(10)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\nNOTE:"
        "\nThis is a descriptive diagnostic, not a causal "
        "measure of why performance changed."
    )

    print(f"\nSaved:\n{output}")

    return combined


def main():
    header("PHARMAAI - LOGISTIC REGRESSION DIAGNOSTICS")

    print(
        "\nThis script analyzes the existing Logistic Regression."
    )

    print(
        "It does NOT retrain the model or change the threshold."
    )

    (
        train_df,
        temporal_df,
        predictors,
        predictions_2025,
        predictions_2026,
        coefficients,
        model,
    ) = load_inputs()

    # Keep a reference so loading the fitted model is explicitly
    # validated even though diagnostics mainly use its saved outputs.
    if model is None:
        raise ValueError("Saved Logistic Regression model could not be loaded.")

    feature_shift = compare_feature_distributions(
        train_df,
        temporal_df,
        predictors,
    )

    ranked_coefficients = inspect_coefficients(
        coefficients
    )

    probability_results = analyze_probability_shift(
        predictions_2025,
        predictions_2026,
    )

    error_results = analyze_2026_errors(
        predictions_2026
    )

    new_controls = analyze_new_2026_controls(
        predictions_2026
    )

    combined = create_combined_diagnostic(
        feature_shift,
        ranked_coefficients,
    )

    summary = {
        "train_rows_2025": len(train_df),
        "temporal_rows_2026_Q1": len(temporal_df),
        "predictor_count": len(predictors),
        "false_negatives_2026_Q1": int(
            (
                (predictions_2026[TARGET] == 1)
                & (
                    predictions_2026[
                        "predicted_label"
                    ] == 0
                )
            ).sum()
        ),
        "false_positives_2026_Q1": int(
            (
                (predictions_2026[TARGET] == 0)
                & (
                    predictions_2026[
                        "predicted_label"
                    ] == 1
                )
            ).sum()
        ),
        "new_controls_2026_Q1": (
            len(new_controls)
            if new_controls is not None
            else None
        ),
        "threshold_changed": False,
        "model_retrained": False,
    }

    summary_path = (
        DIAGNOSTIC_DIR
        / "diagnostic_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    header("DIAGNOSTIC COMPLETE")

    print(
        "\nThe diagnostic files are saved in:"
    )

    print(DIAGNOSTIC_DIR)

    print(
        "\nMost important files:"
        "\n  feature_distribution_shift.csv"
        "\n  coefficients_ranked.csv"
        "\n  probability_distribution_by_class.csv"
        "\n  false_negatives_2026_Q1.csv"
        "\n  new_controls_2026_Q1.csv"
        "\n  feature_shift_coefficient_diagnostic.csv"
        "\n  diagnostic_summary.json"
    )

    print(
        "\nNEXT:"
        "\nReview these diagnostics before deciding whether the "
        "2025-to-2026 performance drop reflects feature shift, "
        "model limitations, or both."
    )


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )
    main()
