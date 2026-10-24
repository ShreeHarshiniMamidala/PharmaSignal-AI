from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PharmaAI - Top-K Ranking Evaluation
# ============================================================
#
# Evaluates:
#   Logistic Regression
#   Random Forest
#   Gradient Boosting
#   ROR
#   PRR
#   IC
#
# at K = 10, 20, 50.
#
# Metrics:
#   Precision@K = positives found in top K / K
#   Recall@K    = positives found in top K / all positives
#   Hits@K      = number of positive controls in top K
#
# ML ranking score:
#   predicted_probability
#
# Traditional ranking scores:
#   ror_score, prr_score, ic_score
#
# IMPORTANT:
# - 2025 ML scores are OOF probabilities.
# - 2026 Q1 ML scores are temporal-test probabilities.
# - Traditional scores come from the already evaluated full-data
#   OMOP control-level tables.
# - No thresholds are created or tuned.
# - All methods are ranked from highest score to lowest score.
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\mamid\OneDrive\Desktop\PharmaAI"
)

RESULTS_ROOT = PROJECT_ROOT / "results"

K_VALUES = [10, 20, 50]

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

OUTPUT_ROOT = RESULTS_ROOT / "top_k_ranking"
TABLE_DIR = OUTPUT_ROOT / "tables"
RANKING_DIR = OUTPUT_ROOT / "ranked_controls"

for folder in [TABLE_DIR, RANKING_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


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


def standardize_prediction_file(path, method, dataset):
    require_file(path)

    df = pd.read_csv(path)

    label_col = find_column(
        df,
        ["groundTruth", "ground_truth", "label", "y_true"],
        "ground-truth column",
    )

    score_col = find_column(
        df,
        [
            "predicted_probability",
            "probability",
            "positive_probability",
            "y_probability",
            "score",
        ],
        "prediction probability/score column",
    )

    key_col = find_column(
        df,
        [
            "reference_control_key",
            "control_key",
        ],
        "reference-control key",
    )

    output = pd.DataFrame({
        "reference_control_key": df[key_col].astype(str),
        "groundTruth": pd.to_numeric(
            df[label_col],
            errors="raise",
        ).astype(int),
        "score": pd.to_numeric(
            df[score_col],
            errors="coerce",
        ),
    })

    # Carry useful identity columns when available.
    for candidate in [
        "omop_drug",
        "omop_outcome",
        "exposureName",
        "outcomeName",
    ]:
        if candidate in df.columns:
            output[candidate] = df[candidate]

    output["dataset"] = dataset
    output["method"] = method

    return output


def load_traditional_file(path, dataset):
    require_file(path)

    df = pd.read_csv(path)

    label_col = find_column(
        df,
        ["groundTruth", "ground_truth"],
        "ground-truth column",
    )

    key_col = find_column(
        df,
        ["reference_control_key", "control_key"],
        "reference-control key",
    )

    metric_columns = {
        "ROR": find_column(
            df,
            ["ror_score", "ror"],
            "ROR score",
        ),
        "PRR": find_column(
            df,
            ["prr_score", "prr"],
            "PRR score",
        ),
        "IC": find_column(
            df,
            ["ic_score", "ic"],
            "IC score",
        ),
    }

    outputs = {}

    for method, score_col in metric_columns.items():
        out = pd.DataFrame({
            "reference_control_key": df[key_col].astype(str),
            "groundTruth": pd.to_numeric(
                df[label_col],
                errors="raise",
            ).astype(int),
            "score": pd.to_numeric(
                df[score_col],
                errors="coerce",
            ),
        })

        for candidate in [
            "exposureName",
            "outcomeName",
            "standardized_outcome",
        ]:
            if candidate in df.columns:
                out[candidate] = df[candidate]

        out["dataset"] = dataset
        out["method"] = method

        outputs[method] = out

    return outputs


def validate_controls(df, method, dataset):
    if df["reference_control_key"].duplicated().any():
        duplicates = int(
            df["reference_control_key"].duplicated().sum()
        )
        raise ValueError(
            f"{method} {dataset} contains "
            f"{duplicates} duplicate control keys."
        )

    labels = set(
        df["groundTruth"].dropna().unique()
    )

    if not labels.issubset({0, 1}):
        raise ValueError(
            f"{method} {dataset} has unexpected labels: "
            f"{sorted(labels)}"
        )

    missing_scores = int(
        df["score"].isna().sum()
    )

    if missing_scores:
        print(
            f"WARNING: {method} {dataset} has "
            f"{missing_scores} missing scores. "
            "Those controls will be excluded for that method."
        )

    return df.dropna(
        subset=["score"]
    ).copy()


def evaluate_top_k(df, method, dataset):
    df = validate_controls(
        df,
        method,
        dataset,
    )

    # Stable sort gives deterministic handling when scores tie.
    ranked = df.sort_values(
        by=["score", "reference_control_key"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    ranked["rank"] = (
        np.arange(len(ranked)) + 1
    )

    total_controls = len(ranked)
    total_positives = int(
        ranked["groundTruth"].sum()
    )
    total_negatives = (
        total_controls - total_positives
    )

    rows = []

    for requested_k in K_VALUES:
        actual_k = min(
            requested_k,
            total_controls,
        )

        top = ranked.head(actual_k)

        hits = int(
            top["groundTruth"].sum()
        )

        precision_at_k = (
            hits / actual_k
            if actual_k > 0
            else np.nan
        )

        recall_at_k = (
            hits / total_positives
            if total_positives > 0
            else np.nan
        )

        rows.append({
            "dataset": dataset,
            "method": method,
            "requested_k": requested_k,
            "actual_k": actual_k,
            "hits_at_k": hits,
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "total_controls": total_controls,
            "total_positives": total_positives,
            "total_negatives": total_negatives,
        })

    return pd.DataFrame(rows), ranked


def check_same_control_universe(method_frames, dataset):
    header(
        f"CHECKING CONTROL UNIVERSE: {dataset}"
    )

    key_sets = {
        method: set(
            df["reference_control_key"].astype(str)
        )
        for method, df in method_frames.items()
    }

    methods = list(key_sets)

    reference_method = methods[0]
    reference_keys = key_sets[
        reference_method
    ]

    all_same = True

    for method in methods[1:]:
        same = (
            key_sets[method]
            == reference_keys
        )

        print(
            f"{reference_method} vs {method}: "
            f"{'SAME' if same else 'DIFFERENT'}"
        )

        if not same:
            all_same = False

    if all_same:
        print(
            "\nAll six methods use the same control universe."
        )
    else:
        print(
            "\nWARNING: control universes differ. "
            "Review the saved control counts before making "
            "a strict top-k comparison."
        )


def save_ranked_file(ranked, method, dataset):
    safe_method = (
        method.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    dataset_folder = (
        RANKING_DIR / dataset
    )

    dataset_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        dataset_folder
        / f"{safe_method}_ranked_controls.csv"
    )

    ranked.to_csv(
        path,
        index=False,
    )


def main():
    header(
        "PHARMAAI - TOP-K RANKING EVALUATION"
    )

    print(
        "\nK values:"
        f"\n  {K_VALUES}"
    )

    print(
        "\nMetrics:"
        "\n  Precision@K"
        "\n  Recall@K"
        "\n  Hits@K"
    )

    print(
        "\nNo classification threshold is used."
    )

    all_results = []

    for dataset in [
        "2025",
        "2026_Q1",
    ]:
        header(
            f"LOADING SCORES: {dataset}"
        )

        method_frames = {}

        # ML methods
        for method, files in ML_FILES.items():
            frame = standardize_prediction_file(
                files[dataset],
                method,
                dataset,
            )

            method_frames[method] = frame

            print(
                f"{method}: "
                f"{len(frame):,} controls"
            )

        # Traditional methods
        traditional = load_traditional_file(
            TRADITIONAL_FILES[dataset],
            dataset,
        )

        for method, frame in traditional.items():
            method_frames[method] = frame

            print(
                f"{method}: "
                f"{len(frame):,} controls"
            )

        check_same_control_universe(
            method_frames,
            dataset,
        )

        header(
            f"TOP-K RESULTS: {dataset}"
        )

        for method, frame in method_frames.items():
            result, ranked = evaluate_top_k(
                frame,
                method,
                dataset,
            )

            all_results.append(
                result
            )

            save_ranked_file(
                ranked,
                method,
                dataset,
            )

            print(
                f"\n{method}"
            )

            for _, row in result.iterrows():
                print(
                    f"  K={int(row['requested_k']):>2}: "
                    f"Hits={int(row['hits_at_k']):>2}, "
                    f"Precision@K={row['precision_at_k']:.4f}, "
                    f"Recall@K={row['recall_at_k']:.4f}"
                )

    final = pd.concat(
        all_results,
        ignore_index=True,
    )

    method_order = [
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting",
        "ROR",
        "PRR",
        "IC",
    ]

    final["method"] = pd.Categorical(
        final["method"],
        categories=method_order,
        ordered=True,
    )

    final = final.sort_values(
        [
            "dataset",
            "requested_k",
            "method",
        ]
    ).reset_index(drop=True)

    final_path = (
        TABLE_DIR
        / "top_k_ranking_results.csv"
    )

    final.to_csv(
        final_path,
        index=False,
    )

    # Separate period tables
    for dataset in [
        "2025",
        "2026_Q1",
    ]:
        subset = final[
            final["dataset"] == dataset
        ]

        subset.to_csv(
            TABLE_DIR
            / f"top_k_ranking_{dataset}.csv",
            index=False,
        )

    # Compact comparison tables: one row per method.
    for dataset in [
        "2025",
        "2026_Q1",
    ]:
        subset = final[
            final["dataset"] == dataset
        ].copy()

        precision_table = subset.pivot(
            index="method",
            columns="requested_k",
            values="precision_at_k",
        ).reset_index()

        precision_table.columns = [
            "method"
            if c == "method"
            else f"precision_at_{int(c)}"
            for c in precision_table.columns
        ]

        recall_table = subset.pivot(
            index="method",
            columns="requested_k",
            values="recall_at_k",
        ).reset_index()

        recall_table.columns = [
            "method"
            if c == "method"
            else f"recall_at_{int(c)}"
            for c in recall_table.columns
        ]

        compact = precision_table.merge(
            recall_table,
            on="method",
            how="inner",
        )

        compact.to_csv(
            TABLE_DIR
            / f"top_k_compact_{dataset}.csv",
            index=False,
        )

    header(
        "TOP-K RANKING EVALUATION COMPLETE"
    )

    print(
        f"\nMain results:\n{final_path}"
    )

    print(
        "\nOutput folders:"
        f"\n  Tables: {TABLE_DIR}"
        f"\n  Ranked controls: {RANKING_DIR}"
    )

    print(
        "\nNext step:"
        "\nUse these results to create the final "
        "Precision@K and Recall@K comparison figures."
    )


if __name__ == "__main__":
    main()
