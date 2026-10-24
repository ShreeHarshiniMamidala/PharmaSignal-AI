from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
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


# ============================================================
# PharmaAI - Gradient Boosting
# ============================================================
#
# PRIMARY DESIGN
# --------------
# 2025:
#   - development/training data
#   - 5-fold stratified cross-validation
#   - ONE primary accuracy is reported from 2025 OOF predictions
#
# 2026 Q1:
#   - later-period temporal test
#   - NOT used for training
#   - accuracy is intentionally NOT reported
#   - precision, recall, F1, AUROC, and PR-AUC are retained
#
# IMPORTANT
# ---------
# ROR, PRR, and IC are excluded from ML predictors.
# Threshold is fixed at 0.50.
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\mamid\OneDrive\Desktop\PharmaAI"
)

ML_DIR = PROJECT_ROOT / "data" / "processed" / "ml"

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "ml"
    / "gradient_boosting"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "gradient_boosting"
)

TRAIN_FILE = ML_DIR / "ml_features_2025.parquet"
TEMPORAL_FILE = ML_DIR / "ml_features_2026_Q1.parquet"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "groundTruth"
CONTROL_KEY = "reference_control_key"
THRESHOLD = 0.50
RANDOM_STATE = 42


def header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def load_data():
    header("LOADING ML FEATURE TABLES")

    require_file(TRAIN_FILE)
    require_file(TEMPORAL_FILE)

    train_df = pd.read_parquet(TRAIN_FILE)
    temporal_df = pd.read_parquet(TEMPORAL_FILE)

    print(f"\n2025 rows: {len(train_df):,}")
    print(f"2026 Q1 rows: {len(temporal_df):,}")

    return train_df, temporal_df


def validate_control_table(df, name):
    header(f"VALIDATING {name}")

    required = {
        TARGET,
        CONTROL_KEY,
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"{name} is missing required columns: "
            f"{sorted(missing)}"
        )

    duplicate_controls = df[CONTROL_KEY].duplicated().sum()

    print(f"\nRows: {len(df):,}")
    print(
        f"Unique controls: "
        f"{df[CONTROL_KEY].nunique():,}"
    )
    print(
        f"Duplicate control rows: "
        f"{duplicate_controls:,}"
    )

    if duplicate_controls > 0:
        raise ValueError(
            f"{name} contains duplicate reference controls. "
            "Expected one row per OMOP control."
        )

    labels = sorted(
        pd.Series(df[TARGET])
        .dropna()
        .unique()
        .tolist()
    )

    print(f"Labels present: {labels}")

    if labels != [0, 1]:
        raise ValueError(
            f"{name} must contain both binary labels 0 and 1."
        )

    print(
        f"Positive controls: "
        f"{(df[TARGET] == 1).sum():,}"
    )
    print(
        f"Negative controls: "
        f"{(df[TARGET] == 0).sum():,}"
    )


def choose_predictors(train_df, temporal_df):
    header("SELECTING GRADIENT BOOSTING PREDICTORS")

    excluded_exact = {
        TARGET,
        CONTROL_KEY,
        "dataset",
        "omop_drug",
        "omop_outcome",
        "omop_source_outcome",
        "exposureId",
        "outcomeId",
        "exposureName",
        "outcomeName",
        "ground_truth",
        "label",
    }

    forbidden_fragments = (
        "ror",
        "prr",
        "continuity_correction",
    )

    def forbidden(column):
        c = column.lower()

        if any(
            fragment in c
            for fragment in forbidden_fragments
        ):
            return True

        if (
            c == "ic"
            or c.startswith("ic_")
            or c.endswith("_ic")
            or "_ic_" in c
        ):
            return True

        return False

    shared_columns = [
        c
        for c in train_df.columns
        if c in temporal_df.columns
    ]

    predictors = []

    for column in shared_columns:
        if column in excluded_exact:
            continue

        if forbidden(column):
            continue

        if not pd.api.types.is_numeric_dtype(
            train_df[column]
        ):
            continue

        if not pd.api.types.is_numeric_dtype(
            temporal_df[column]
        ):
            continue

        predictors.append(column)

    if not predictors:
        raise ValueError(
            "No numeric predictors were selected."
        )

    print(f"\nPredictor count: {len(predictors):,}")

    print("\nPredictors:")
    for predictor in predictors:
        print(f"  - {predictor}")

    forbidden_selected = [
        c
        for c in predictors
        if forbidden(c)
    ]

    if forbidden_selected:
        raise ValueError(
            "Traditional disproportionality metrics were "
            f"selected accidentally: {forbidden_selected}"
        )

    print("\nROR used as ML feature: False")
    print("PRR used as ML feature: False")
    print("IC used as ML feature: False")

    return predictors


def validate_predictors(df, predictors, name):
    header(f"CHECKING {name} PREDICTORS")

    x = df[predictors].copy()

    missing_count = int(
        x.isna().sum().sum()
    )

    numeric = x.to_numpy(
        dtype=float
    )

    infinite_count = int(
        np.isinf(numeric).sum()
    )

    print(
        f"\nMissing predictor cells: "
        f"{missing_count:,}"
    )
    print(
        f"Infinite predictor cells: "
        f"{infinite_count:,}"
    )

    if missing_count > 0:
        raise ValueError(
            f"{name} contains missing predictor values."
        )

    if infinite_count > 0:
        raise ValueError(
            f"{name} contains infinite predictor values."
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
        "n": int(len(y_true)),
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


def make_model():
    # Standard sklearn Gradient Boosting.
    # Conservative/default-style settings are used so we do not
    # tune against the 2026 temporal test set.
    return GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=RANDOM_STATE,
    )


def run_2025_cross_validation(
    train_df,
    predictors,
):
    header("2025 - 5-FOLD STRATIFIED CROSS-VALIDATION")

    x_2025 = train_df[predictors]
    y_2025 = train_df[TARGET].astype(int)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    model = make_model()

    print(
        "\nGenerating out-of-fold probabilities..."
    )

    oof_probabilities = cross_val_predict(
        model,
        x_2025,
        y_2025,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    metrics = calculate_metrics(
        y_2025,
        oof_probabilities,
    )

    predictions = (
        oof_probabilities >= THRESHOLD
    ).astype(int)

    # ONE PRIMARY ACCURACY ONLY.
    primary_accuracy = accuracy_score(
        y_2025,
        predictions,
    )

    metrics["accuracy"] = primary_accuracy

    print(
        "\nPRIMARY GRADIENT BOOSTING ACCURACY "
        "(2025 5-fold stratified OOF): "
        f"{primary_accuracy:.4f} "
        f"({primary_accuracy * 100:.2f}%)"
    )

    print("\nOther 2025 metrics:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  AUROC:     {metrics['auroc']:.4f}")
    print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
    print(
        "  Confusion matrix: "
        f"TN={metrics['tn']}, "
        f"FP={metrics['fp']}, "
        f"FN={metrics['fn']}, "
        f"TP={metrics['tp']}"
    )

    prediction_df = train_df[
        [
            c
            for c in [
                CONTROL_KEY,
                "omop_drug",
                "omop_outcome",
                TARGET,
            ]
            if c in train_df.columns
        ]
    ].copy()

    prediction_df[
        "predicted_probability"
    ] = oof_probabilities

    prediction_df[
        "predicted_label"
    ] = predictions

    prediction_path = (
        RESULTS_DIR
        / "gradient_boosting_predictions_2025_oof.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    print(
        f"\n2025 OOF predictions saved:\n"
        f"{prediction_path}"
    )

    return metrics


def check_temporal_overlap(
    train_df,
    temporal_df,
):
    header("CHECKING 2025 / 2026 Q1 CONTROL OVERLAP")

    train_keys = set(
        train_df[CONTROL_KEY].astype(str)
    )

    temporal_keys = set(
        temporal_df[CONTROL_KEY].astype(str)
    )

    overlap = train_keys.intersection(
        temporal_keys
    )

    new_2026 = temporal_keys.difference(
        train_keys
    )

    summary = pd.DataFrame(
        [
            {
                "train_controls_2025": len(train_keys),
                "test_controls_2026_Q1": len(temporal_keys),
                "overlap_controls": len(overlap),
                "new_controls_2026_Q1": len(new_2026),
                "overlap_fraction_of_2026": (
                    len(overlap)
                    / len(temporal_keys)
                    if temporal_keys
                    else np.nan
                ),
            }
        ]
    )

    output = (
        RESULTS_DIR
        / "temporal_overlap_summary.csv"
    )

    summary.to_csv(
        output,
        index=False,
    )

    print(
        f"\n2025 controls: {len(train_keys):,}"
    )
    print(
        f"2026 Q1 controls: {len(temporal_keys):,}"
    )
    print(
        f"Controls appearing in both periods: "
        f"{len(overlap):,}"
    )
    print(
        f"Controls new in 2026 Q1: "
        f"{len(new_2026):,}"
    )

    if temporal_keys:
        print(
            "2026 Q1 overlap percentage: "
            f"{len(overlap) / len(temporal_keys):.2%}"
        )

    print(f"\nSaved:\n{output}")


def train_final_model(
    train_df,
    predictors,
):
    header("TRAINING FINAL GRADIENT BOOSTING ON ALL 2025 DATA")

    x_2025 = train_df[predictors]
    y_2025 = train_df[TARGET].astype(int)

    model = make_model()

    model.fit(
        x_2025,
        y_2025,
    )

    model_path = (
        MODEL_DIR
        / "gradient_boosting_2025.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"\nFinal model saved:\n{model_path}"
    )

    return model


def save_feature_importance(
    model,
    predictors,
):
    header("SAVING GRADIENT BOOSTING FEATURE IMPORTANCE")

    importance_df = pd.DataFrame(
        {
            "feature": predictors,
            "importance": model.feature_importances_,
        }
    )

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    output = (
        RESULTS_DIR
        / "gradient_boosting_feature_importance.csv"
    )

    importance_df.to_csv(
        output,
        index=False,
    )

    print("\nTop 10 features:")
    print(
        importance_df.head(10).to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nSaved:\n{output}")


def evaluate_2026(
    model,
    temporal_df,
    predictors,
    train_df,
):
    header("2026 Q1 - TEMPORAL TEST")

    x_2026 = temporal_df[predictors]
    y_2026 = temporal_df[TARGET].astype(int)

    probabilities = model.predict_proba(
        x_2026
    )[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    metrics = calculate_metrics(
        y_2026,
        probabilities,
    )

    # Accuracy is deliberately not calculated/reported here.
    print(
        "\n2026 Q1 temporal metrics "
        "(accuracy intentionally not reported):"
    )
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  AUROC:     {metrics['auroc']:.4f}")
    print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
    print(
        "  Confusion matrix: "
        f"TN={metrics['tn']}, "
        f"FP={metrics['fp']}, "
        f"FN={metrics['fn']}, "
        f"TP={metrics['tp']}"
    )

    prediction_columns = [
        c
        for c in [
            CONTROL_KEY,
            "omop_drug",
            "omop_outcome",
            TARGET,
        ]
        if c in temporal_df.columns
    ]

    prediction_df = temporal_df[
        prediction_columns
    ].copy()

    train_keys = set(
        train_df[CONTROL_KEY].astype(str)
    )

    prediction_df[
        "seen_in_2025"
    ] = temporal_df[
        CONTROL_KEY
    ].astype(str).isin(
        train_keys
    ).to_numpy()

    prediction_df[
        "predicted_probability"
    ] = probabilities

    prediction_df[
        "predicted_label"
    ] = predictions

    output = (
        RESULTS_DIR
        / "gradient_boosting_predictions_2026_Q1.csv"
    )

    prediction_df.to_csv(
        output,
        index=False,
    )

    print(
        f"\n2026 Q1 predictions saved:\n{output}"
    )

    return metrics


def save_results(
    cv_metrics,
    temporal_metrics,
    predictors,
):
    header("SAVING GRADIENT BOOSTING RESULTS")

    # Save exactly ONE primary accuracy.
    primary_accuracy_path = (
        RESULTS_DIR
        / "gradient_boosting_primary_accuracy.csv"
    )

    pd.DataFrame(
        [
            {
                "model": "Gradient Boosting",
                "evaluation": "2025_5fold_stratified_oof",
                "accuracy": cv_metrics["accuracy"],
            }
        ]
    ).to_csv(
        primary_accuracy_path,
        index=False,
    )

    # Save other metrics for both evaluations.
    # Accuracy is intentionally excluded.
    rows = []

    cv_row = {
        "model": "Gradient Boosting",
        "evaluation": "2025_5fold_stratified_oof",
    }

    for key, value in cv_metrics.items():
        if key != "accuracy":
            cv_row[key] = value

    rows.append(cv_row)

    temporal_row = {
        "model": "Gradient Boosting",
        "evaluation": "2026_Q1_temporal_test",
        **temporal_metrics,
    }

    rows.append(temporal_row)

    metrics_df = pd.DataFrame(rows)

    metrics_path = (
        RESULTS_DIR
        / "gradient_boosting_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    predictor_path = (
        RESULTS_DIR
        / "gradient_boosting_predictors.json"
    )

    predictor_path.write_text(
        json.dumps(
            predictors,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\nPrimary accuracy saved:\n"
        f"{primary_accuracy_path}"
    )

    print(
        f"\nOther metrics saved:\n"
        f"{metrics_path}"
    )

    print(
        f"\nPredictor list saved:\n"
        f"{predictor_path}"
    )


def main():
    header("PHARMAAI - GRADIENT BOOSTING")

    print(
        "\nPrimary experiment:"
        "\n  Train/develop on 2025"
        "\n  Test temporally on 2026 Q1"
        "\n  Report ONE primary accuracy from 2025 5-fold OOF CV"
    )

    train_df, temporal_df = load_data()

    validate_control_table(
        train_df,
        "2025",
    )

    validate_control_table(
        temporal_df,
        "2026 Q1",
    )

    predictors = choose_predictors(
        train_df,
        temporal_df,
    )

    validate_predictors(
        train_df,
        predictors,
        "2025",
    )

    validate_predictors(
        temporal_df,
        predictors,
        "2026 Q1",
    )

    cv_metrics = run_2025_cross_validation(
        train_df,
        predictors,
    )

    check_temporal_overlap(
        train_df,
        temporal_df,
    )

    final_model = train_final_model(
        train_df,
        predictors,
    )

    save_feature_importance(
        final_model,
        predictors,
    )

    temporal_metrics = evaluate_2026(
        final_model,
        temporal_df,
        predictors,
        train_df,
    )

    save_results(
        cv_metrics,
        temporal_metrics,
        predictors,
    )

    header("GRADIENT BOOSTING COMPLETE")

    print(
        "\nPrimary Gradient Boosting accuracy:"
        f"\n  {cv_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        "\nRemember:"
        "\n  2025 = training/development"
        "\n  2026 Q1 = temporal testing"
        "\n  2026 accuracy is intentionally not reported"
    )

    print(
        "\nNEXT:"
        "\nReview Gradient Boosting results, then compare "
        "the three ML models."
    )


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )
    main()
