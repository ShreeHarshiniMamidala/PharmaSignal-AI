from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# PharmaAI - Logistic Regression
# ============================================================
#
# EXPERIMENT
# ----------
# 2025:
#   Development data
#   5-fold stratified cross-validation
#
# 2026 Q1:
#   Temporal evaluation data
#
# TARGET
# ------
# groundTruth
#   1 = OMOP positive control
#   0 = OMOP negative control
#
# IMPORTANT
# ---------
# ROR, PRR, and IC are NOT used as Logistic Regression
# predictors. They remain separate traditional baselines.
#
# The input feature files already contain one row per matched
# OMOP drug-outcome control.
# ============================================================


RANDOM_STATE = 42
N_SPLITS = 5
THRESHOLD = 0.50

PROJECT_ROOT = Path(
    r"C:\Users\mamid\OneDrive\Desktop\PharmaAI"
)

ML_DIR = PROJECT_ROOT / "data" / "processed" / "ml"

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "ml"
    / "logistic_regression"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "logistic_regression"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = ML_DIR / "ml_features_2025.parquet"
TEMPORAL_FILE = ML_DIR / "ml_features_2026_Q1.parquet"

TARGET = "groundTruth"

NON_PREDICTOR_COLUMNS = {
    "reference_control_key",
    "omop_drug",
    "omop_outcome",
    "omop_outcome_source",
    "groundTruth",
    "dataset",
}

FORBIDDEN_FEATURE_TOKENS = (
    "ror",
    "prr",
    "ic",
    "continuity_correction",
)


def header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_feature_file(path, dataset_name):
    header(f"LOADING {dataset_name}")

    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n{path}\n\n"
            "Run build_ml_features.py first."
        )

    df = pd.read_parquet(path)

    required = {
        "reference_control_key",
        "omop_drug",
        "omop_outcome",
        TARGET,
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{sorted(missing)}"
        )

    if df["reference_control_key"].duplicated().any():
        raise ValueError(
            f"{dataset_name} contains duplicate OMOP controls."
        )

    labels = set(
        pd.to_numeric(
            df[TARGET],
            errors="coerce",
        ).dropna().astype(int).unique()
    )

    if labels != {0, 1}:
        raise ValueError(
            f"{dataset_name}: groundTruth must contain both 0 and 1. "
            f"Found: {sorted(labels)}"
        )

    df[TARGET] = df[TARGET].astype(int)

    print(f"\nRows: {len(df):,}")
    print(f"Positive controls: {(df[TARGET] == 1).sum():,}")
    print(f"Negative controls: {(df[TARGET] == 0).sum():,}")

    return df


def select_predictors(train_df, temporal_df):
    header("SELECTING LOGISTIC REGRESSION PREDICTORS")

    shared_columns = [
        column
        for column in train_df.columns
        if column in temporal_df.columns
    ]

    predictors = []

    for column in shared_columns:
        if column in NON_PREDICTOR_COLUMNS:
            continue

        lower_name = column.lower()

        if any(
            token in lower_name
            for token in FORBIDDEN_FEATURE_TOKENS
        ):
            continue

        if pd.api.types.is_numeric_dtype(train_df[column]):
            predictors.append(column)

    if not predictors:
        raise ValueError("No numeric predictors were found.")

    print(f"\nPredictor count: {len(predictors)}")

    for column in predictors:
        print(f"  - {column}")

    print("\nTraditional metrics used as predictors:")
    print("  ROR: False")
    print("  PRR: False")
    print("  IC:  False")

    return predictors


def validate_predictors(train_df, temporal_df, predictors):
    header("VALIDATING PREDICTORS")

    train_missing = int(
        train_df[predictors].isna().sum().sum()
    )

    temporal_missing = int(
        temporal_df[predictors].isna().sum().sum()
    )

    train_inf = int(
        np.isinf(
            train_df[predictors].to_numpy(dtype=float)
        ).sum()
    )

    temporal_inf = int(
        np.isinf(
            temporal_df[predictors].to_numpy(dtype=float)
        ).sum()
    )

    print(f"\n2025 missing predictor cells: {train_missing:,}")
    print(f"2026 Q1 missing predictor cells: {temporal_missing:,}")
    print(f"2025 infinite predictor cells: {train_inf:,}")
    print(f"2026 Q1 infinite predictor cells: {temporal_inf:,}")

    if train_missing or temporal_missing:
        raise ValueError(
            "Missing predictor values were found. "
            "Inspect the feature-building stage before training."
        )

    if train_inf or temporal_inf:
        raise ValueError(
            "Infinite predictor values were found. "
            "Inspect the feature-building stage before training."
        )

    print("\nPredictor validation: PASS")


def build_pipeline():
    # StandardScaler is important for Logistic Regression because
    # the count/rate features are on very different numeric scales.
    #
    # class_weight='balanced' gives more weight to the smaller class.
    # This is useful here because the reference controls are not
    # perfectly balanced.

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "logistic_regression",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    solver="liblinear",
                ),
            ),
        ]
    )


def calculate_metrics(y_true, probabilities):
    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "n": len(y_true),
        "positive_n": int((y_true == 1).sum()),
        "negative_n": int((y_true == 0).sum()),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "auroc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_predictions(
    source_df,
    probabilities,
    output_path,
    seen_in_2025=None,
):
    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    columns = [
        "reference_control_key",
        "omop_drug",
        "omop_outcome",
        TARGET,
    ]

    output = source_df[columns].copy()

    output["predicted_probability"] = probabilities
    output["predicted_label"] = predictions

    if seen_in_2025 is not None:
        output["seen_in_2025"] = seen_in_2025

    output.to_csv(
        output_path,
        index=False,
    )


def main():
    header("PHARMAAI - LOGISTIC REGRESSION")

    print("\nExperimental design:")
    print("  2025 = development dataset")
    print("  5-fold stratified CV = development evaluation")
    print("  2026 Q1 = temporal evaluation")
    print("  Classification threshold = 0.50")
    print("  ROR/PRR/IC = excluded from ML predictors")

    train_df = load_feature_file(
        TRAIN_FILE,
        "2025 ML FEATURES",
    )

    temporal_df = load_feature_file(
        TEMPORAL_FILE,
        "2026 Q1 ML FEATURES",
    )

    predictors = select_predictors(
        train_df,
        temporal_df,
    )

    validate_predictors(
        train_df,
        temporal_df,
        predictors,
    )

    predictor_path = (
        RESULTS_DIR
        / "logistic_regression_predictors.json"
    )

    predictor_path.write_text(
        json.dumps(
            predictors,
            indent=2,
        ),
        encoding="utf-8",
    )

    X_2025 = train_df[predictors].copy()
    y_2025 = train_df[TARGET].copy()

    X_2026 = temporal_df[predictors].copy()
    y_2026 = temporal_df[TARGET].copy()

    # ========================================================
    # Check overlap between the two time periods
    # ========================================================

    header("CHECKING TEMPORAL CONTROL OVERLAP")

    train_keys = set(
        train_df["reference_control_key"].astype(str)
    )

    temporal_keys = set(
        temporal_df["reference_control_key"].astype(str)
    )

    overlap_keys = train_keys.intersection(
        temporal_keys
    )

    new_2026_keys = temporal_keys.difference(
        train_keys
    )

    print(f"\n2025 controls: {len(train_keys):,}")
    print(f"2026 Q1 controls: {len(temporal_keys):,}")
    print(
        "Present in both periods: "
        f"{len(overlap_keys):,}"
    )
    print(
        "New in 2026 Q1 relative to matched 2025 set: "
        f"{len(new_2026_keys):,}"
    )

    overlap_summary = pd.DataFrame(
        [
            {
                "controls_2025": len(train_keys),
                "controls_2026_Q1": len(temporal_keys),
                "overlap_controls": len(overlap_keys),
                "new_2026_controls": len(new_2026_keys),
                "overlap_fraction_of_2026": (
                    len(overlap_keys)
                    / len(temporal_keys)
                ),
            }
        ]
    )

    overlap_summary.to_csv(
        RESULTS_DIR
        / "temporal_overlap_summary.csv",
        index=False,
    )

    # ========================================================
    # 2025: 5-fold stratified cross-validation
    # ========================================================

    header("2025 - 5-FOLD STRATIFIED CROSS-VALIDATION")

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline()

    # Out-of-fold prediction:
    # each control is predicted by a model that did NOT train
    # on that control.
    oof_probabilities = cross_val_predict(
        pipeline,
        X_2025,
        y_2025,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    cv_metrics = calculate_metrics(
        y_2025,
        oof_probabilities,
    )

    # PRIMARY ACCURACY:
    # Accuracy is reported only for the 2025 5-fold stratified
    # out-of-fold evaluation. Temporal evaluations retain the
    # other metrics but do not calculate/report additional accuracies.
    cv_metrics["accuracy"] = accuracy_score(
        y_2025,
        (oof_probabilities >= THRESHOLD).astype(int),
    )

    print("\n2025 out-of-fold results:")
    print(f"  Accuracy:  {cv_metrics['accuracy']:.4f}")
    print(f"  Precision: {cv_metrics['precision']:.4f}")
    print(f"  Recall:    {cv_metrics['recall']:.4f}")
    print(f"  F1:        {cv_metrics['f1']:.4f}")
    print(f"  AUROC:     {cv_metrics['auroc']:.4f}")
    print(f"  PR-AUC:    {cv_metrics['pr_auc']:.4f}")

    print("\nConfusion matrix counts:")
    print(f"  TN: {cv_metrics['tn']}")
    print(f"  FP: {cv_metrics['fp']}")
    print(f"  FN: {cv_metrics['fn']}")
    print(f"  TP: {cv_metrics['tp']}")

    oof_path = (
        RESULTS_DIR
        / "logistic_regression_predictions_2025_oof.csv"
    )

    save_predictions(
        train_df,
        oof_probabilities,
        oof_path,
    )

    # ========================================================
    # Train final Logistic Regression on ALL 2025 controls
    # ========================================================

    header("TRAINING FINAL LOGISTIC REGRESSION ON ALL 2025 CONTROLS")

    final_model = build_pipeline()

    final_model.fit(
        X_2025,
        y_2025,
    )

    model_path = (
        MODEL_DIR
        / "logistic_regression_2025.joblib"
    )

    joblib.dump(
        final_model,
        model_path,
    )

    print(f"\nModel saved:\n{model_path}")

    # ========================================================
    # Coefficients
    # ========================================================

    logistic_model = (
        final_model
        .named_steps[
            "logistic_regression"
        ]
    )

    coefficient_df = pd.DataFrame(
        {
            "feature": predictors,
            "coefficient": logistic_model.coef_[0],
        }
    )

    coefficient_df[
        "absolute_coefficient"
    ] = coefficient_df[
        "coefficient"
    ].abs()

    coefficient_df = (
        coefficient_df
        .sort_values(
            "absolute_coefficient",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    coefficient_path = (
        RESULTS_DIR
        / "logistic_regression_coefficients.csv"
    )

    coefficient_df.to_csv(
        coefficient_path,
        index=False,
    )

    print(
        "\nCoefficient table saved:\n"
        f"{coefficient_path}"
    )

    # ========================================================
    # 2026 Q1 temporal evaluation
    # ========================================================

    header("2026 Q1 - TEMPORAL EVALUATION")

    temporal_probabilities = (
        final_model.predict_proba(
            X_2026
        )[:, 1]
    )

    temporal_metrics = calculate_metrics(
        y_2026,
        temporal_probabilities,
    )

    print("\n2026 Q1 results:")
    print(
        f"  Precision: "
        f"{temporal_metrics['precision']:.4f}"
    )
    print(
        f"  Recall:    "
        f"{temporal_metrics['recall']:.4f}"
    )
    print(
        f"  F1:        "
        f"{temporal_metrics['f1']:.4f}"
    )
    print(
        f"  AUROC:     "
        f"{temporal_metrics['auroc']:.4f}"
    )
    print(
        f"  PR-AUC:    "
        f"{temporal_metrics['pr_auc']:.4f}"
    )

    print("\nConfusion matrix counts:")
    print(f"  TN: {temporal_metrics['tn']}")
    print(f"  FP: {temporal_metrics['fp']}")
    print(f"  FN: {temporal_metrics['fn']}")
    print(f"  TP: {temporal_metrics['tp']}")

    seen_in_2025 = (
        temporal_df[
            "reference_control_key"
        ]
        .astype(str)
        .isin(overlap_keys)
        .to_numpy()
    )

    temporal_prediction_path = (
        RESULTS_DIR
        / "logistic_regression_predictions_2026_Q1.csv"
    )

    save_predictions(
        temporal_df,
        temporal_probabilities,
        temporal_prediction_path,
        seen_in_2025=seen_in_2025,
    )

    # ========================================================
    # Optional descriptive evaluation:
    # 2026 controls not represented in the matched 2025 set
    # ========================================================

    new_mask = ~seen_in_2025

    new_control_metrics = None

    if new_mask.sum() > 0:
        y_new = y_2026.loc[new_mask]

        print("\n2026 Q1 new-controls-only subset:")
        print(f"  Rows: {len(y_new):,}")
        print(f"  Positive: {(y_new == 1).sum():,}")
        print(f"  Negative: {(y_new == 0).sum():,}")

        if y_new.nunique() == 2:
            new_control_metrics = calculate_metrics(
                y_new,
                temporal_probabilities[new_mask],
            )

            print(
                f"  AUROC:  "
                f"{new_control_metrics['auroc']:.4f}"
            )
            print(
                f"  PR-AUC: "
                f"{new_control_metrics['pr_auc']:.4f}"
            )
            print(
                f"  F1:     "
                f"{new_control_metrics['f1']:.4f}"
            )
        else:
            print(
                "  AUROC/PR-AUC not calculated because "
                "this subset contains only one class."
            )

    # ========================================================
    # Save metric summary
    # ========================================================

    metric_rows = []

    cv_row = {
        "model": "Logistic Regression",
        "evaluation": "2025_5fold_stratified_oof",
        **cv_metrics,
    }

    temporal_row = {
        "model": "Logistic Regression",
        "evaluation": "2026_Q1_temporal_all",
        **temporal_metrics,
    }

    metric_rows.append(cv_row)
    metric_rows.append(temporal_row)

    if new_control_metrics is not None:
        metric_rows.append(
            {
                "model": "Logistic Regression",
                "evaluation": (
                    "2026_Q1_temporal_new_controls_only"
                ),
                **new_control_metrics,
            }
        )

    metrics_df = pd.DataFrame(
        metric_rows
    )

    metrics_path = (
        RESULTS_DIR
        / "logistic_regression_metrics.csv"
    )

    # Save the single primary accuracy separately so there is no
    # confusing second/third accuracy in the multi-evaluation metrics file.
    primary_accuracy_path = (
        RESULTS_DIR
        / "logistic_regression_primary_accuracy.csv"
    )

    pd.DataFrame(
        [
            {
                "model": "Logistic Regression",
                "evaluation": "2025_5fold_stratified_oof",
                "accuracy": cv_metrics["accuracy"],
            }
        ]
    ).to_csv(
        primary_accuracy_path,
        index=False,
    )

    # Keep precision/recall/F1/AUROC/PR-AUC for all evaluations,
    # but remove accuracy from this table.
    metrics_df.drop(
        columns=["accuracy"],
        errors="ignore",
    ).to_csv(
        metrics_path,
        index=False,
    )

    # ========================================================
    # Final summary
    # ========================================================

    header("LOGISTIC REGRESSION SUMMARY")

    print(
        f"\nPRIMARY LOGISTIC REGRESSION ACCURACY "
        f"(2025 5-fold stratified OOF): "
        f"{cv_metrics['accuracy']:.4f} "
        f"({cv_metrics['accuracy'] * 100:.2f}%)"
    )

    print(
        "\nOther evaluation metrics "
        "(accuracy intentionally not reported):"
    )

    print(
        "\n"
        + metrics_df[
            [
                "evaluation",
                "n",
                "positive_n",
                "negative_n",
                "precision",
                "recall",
                "f1",
                "auroc",
                "pr_auc",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nSaved outputs:")
    print(f"  Metrics: {metrics_path}")
    print(f"  2025 predictions: {oof_path}")
    print(
        "  2026 predictions: "
        f"{temporal_prediction_path}"
    )
    print(f"  Coefficients: {coefficient_path}")
    print(f"  Predictor list: {predictor_path}")
    print(f"  Model: {model_path}")

    header("LOGISTIC REGRESSION COMPLETE")



if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )
    main()
