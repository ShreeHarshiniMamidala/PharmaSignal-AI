from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PharmaAI - Final ML vs Traditional Comparison
# ============================================================
#
# PURPOSE
# -------
# Create final comparison tables and publication/poster-ready
# figures for:
#
#   Logistic Regression
#   Random Forest
#   Gradient Boosting
#   ROR
#   PRR
#   IC
#
# IMPORTANT
# ---------
# 1. The single "primary accuracy" for ML models is the 2025
#    5-fold stratified out-of-fold (OOF) accuracy.
# 2. Traditional methods are continuous ranking scores in this
#    experiment, so this script DOES NOT invent an accuracy for
#    ROR/PRR/IC.
# 3. The fairest common comparison across all six methods is
#    AUROC and PR-AUC.
# 4. 2026 Q1 is shown as later-period temporal evaluation.
# 5. Figures are saved into SEPARATE folders by comparison type.
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\mamid\OneDrive\Desktop\PharmaAI"
)

RESULTS_ROOT = PROJECT_ROOT / "results"

TRADITIONAL_EVAL_FILE = (
    RESULTS_ROOT
    / "traditional_full"
    / "evaluation"
    / "traditional_baseline_evaluation.csv"
)

COMPARISON_ROOT = RESULTS_ROOT / "final_comparison"

TABLE_DIR = COMPARISON_ROOT / "tables"
FIGURE_ROOT = COMPARISON_ROOT / "figures"

FIG_2025_AUROC = FIGURE_ROOT / "01_2025_auroc"
FIG_2025_PRAUC = FIGURE_ROOT / "02_2025_pr_auc"
FIG_2025_COMBINED = FIGURE_ROOT / "03_2025_combined"
FIG_2026_AUROC = FIGURE_ROOT / "04_2026_Q1_auroc"
FIG_2026_PRAUC = FIGURE_ROOT / "05_2026_Q1_pr_auc"
FIG_2026_COMBINED = FIGURE_ROOT / "06_2026_Q1_combined"
FIG_ML_ACCURACY = FIGURE_ROOT / "07_ml_primary_accuracy"
FIG_ML_TEMPORAL = FIGURE_ROOT / "08_ml_temporal_change"

ALL_DIRS = [
    TABLE_DIR,
    FIG_2025_AUROC,
    FIG_2025_PRAUC,
    FIG_2025_COMBINED,
    FIG_2026_AUROC,
    FIG_2026_PRAUC,
    FIG_2026_COMBINED,
    FIG_ML_ACCURACY,
    FIG_ML_TEMPORAL,
]

for directory in ALL_DIRS:
    directory.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Validated ML results from the completed PharmaAI experiments.
# These are entered explicitly so the final comparison preserves
# the exact reported OOF/temporal results.
# ------------------------------------------------------------

ML_RESULTS = pd.DataFrame(
    [
        {
            "dataset": "2025",
            "method": "Logistic Regression",
            "method_group": "Machine Learning",
            "accuracy": 0.6913,
            "precision": 0.8310,
            "recall": 0.6344,
            "f1": 0.7195,
            "auroc": 0.8024,
            "pr_auc": 0.8882,
            "n_controls": 149,
            "positive_controls": 93,
            "negative_controls": 56,
        },
        {
            "dataset": "2025",
            "method": "Random Forest",
            "method_group": "Machine Learning",
            "accuracy": 0.7785,
            "precision": 0.8191,
            "recall": 0.8280,
            "f1": 0.8235,
            "auroc": 0.8337,
            "pr_auc": 0.9028,
            "n_controls": 149,
            "positive_controls": 93,
            "negative_controls": 56,
        },
        {
            "dataset": "2025",
            "method": "Gradient Boosting",
            "method_group": "Machine Learning",
            "accuracy": 0.7718,
            "precision": 0.8315,
            "recall": 0.7957,
            "f1": 0.8132,
            "auroc": 0.8110,
            "pr_auc": 0.8888,
            "n_controls": 149,
            "positive_controls": 93,
            "negative_controls": 56,
        },
        {
            "dataset": "2026_Q1",
            "method": "Logistic Regression",
            "method_group": "Machine Learning",
            "accuracy": np.nan,
            "precision": 0.9375,
            "recall": 0.2000,
            "f1": 0.3297,
            "auroc": 0.6452,
            "pr_auc": 0.8519,
            "n_controls": 103,
            "positive_controls": 75,
            "negative_controls": 28,
        },
        {
            "dataset": "2026_Q1",
            "method": "Random Forest",
            "method_group": "Machine Learning",
            "accuracy": np.nan,
            "precision": 0.8235,
            "recall": 0.5600,
            "f1": 0.6667,
            "auroc": 0.5990,
            "pr_auc": 0.8121,
            "n_controls": 103,
            "positive_controls": 75,
            "negative_controls": 28,
        },
        {
            "dataset": "2026_Q1",
            "method": "Gradient Boosting",
            "method_group": "Machine Learning",
            "accuracy": np.nan,
            "precision": 0.8125,
            "recall": 0.8667,
            "f1": 0.8387,
            "auroc": 0.7438,
            "pr_auc": 0.8846,
            "n_controls": 103,
            "positive_controls": 75,
            "negative_controls": 28,
        },
    ]
)


METHOD_ORDER = [
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
    "ROR",
    "PRR",
    "IC",
]


def header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def load_traditional_results():
    header("LOADING TRADITIONAL BASELINE RESULTS")

    require_file(TRADITIONAL_EVAL_FILE)

    traditional = pd.read_csv(
        TRADITIONAL_EVAL_FILE
    )

    required = {
        "dataset",
        "method",
        "n_controls",
        "positive_controls",
        "negative_controls",
        "auroc",
        "pr_auc",
    }

    missing = required - set(traditional.columns)

    if missing:
        raise ValueError(
            "Traditional evaluation file is missing columns: "
            f"{sorted(missing)}"
        )

    traditional = traditional[
        list(required)
    ].copy()

    traditional["method_group"] = "Traditional"
    traditional["accuracy"] = np.nan
    traditional["precision"] = np.nan
    traditional["recall"] = np.nan
    traditional["f1"] = np.nan

    traditional = traditional[
        [
            "dataset",
            "method",
            "method_group",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auroc",
            "pr_auc",
            "n_controls",
            "positive_controls",
            "negative_controls",
        ]
    ]

    print(
        f"\nTraditional rows loaded: {len(traditional):,}"
    )

    return traditional


def build_master_table(traditional):
    header("BUILDING MASTER COMPARISON TABLE")

    master = pd.concat(
        [
            ML_RESULTS.copy(),
            traditional.copy(),
        ],
        ignore_index=True,
    )

    master["method"] = pd.Categorical(
        master["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )

    master = master.sort_values(
        ["dataset", "method"]
    ).reset_index(drop=True)

    # Validate that every period has all six methods.
    for dataset in ["2025", "2026_Q1"]:
        subset = master[
            master["dataset"] == dataset
        ]

        found = set(
            subset["method"].astype(str)
        )

        missing = set(METHOD_ORDER) - found

        if missing:
            raise ValueError(
                f"{dataset} is missing methods: "
                f"{sorted(missing)}"
            )

    master_path = (
        TABLE_DIR
        / "final_method_comparison.csv"
    )

    master.to_csv(
        master_path,
        index=False,
    )

    print(
        f"\nMaster comparison table saved:\n"
        f"{master_path}"
    )

    return master


def build_common_metric_table(master):
    # AUROC and PR-AUC are available for all six methods and
    # therefore form the direct common comparison.
    common = master[
        [
            "dataset",
            "method",
            "method_group",
            "n_controls",
            "positive_controls",
            "negative_controls",
            "auroc",
            "pr_auc",
        ]
    ].copy()

    path = (
        TABLE_DIR
        / "common_ranking_metrics.csv"
    )

    common.to_csv(
        path,
        index=False,
    )

    return common


def build_ml_only_table(master):
    ml = master[
        master["method_group"]
        == "Machine Learning"
    ].copy()

    path = (
        TABLE_DIR
        / "ml_detailed_metrics.csv"
    )

    ml.to_csv(
        path,
        index=False,
    )

    return ml


def save_bar_chart(
    df,
    metric,
    title,
    ylabel,
    output_path,
    baseline=None,
):
    plot_df = (
        df[
            ["method", metric]
        ]
        .dropna()
        .copy()
    )

    plot_df["method"] = (
        plot_df["method"]
        .astype(str)
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    bars = ax.bar(
        plot_df["method"],
        plot_df[metric],
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Method")
    ax.set_ylim(0, 1)

    ax.tick_params(
        axis="x",
        rotation=25,
    )

    if baseline is not None:
        ax.axhline(
            baseline,
            linestyle="--",
            linewidth=1,
        )

    for bar, value in zip(
        bars,
        plot_df[metric],
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.015,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_grouped_ranking_chart(
    df,
    title,
    output_path,
):
    plot_df = df[
        [
            "method",
            "auroc",
            "pr_auc",
        ]
    ].dropna().copy()

    plot_df["method"] = (
        plot_df["method"]
        .astype(str)
    )

    x = np.arange(
        len(plot_df)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    bars1 = ax.bar(
        x - width / 2,
        plot_df["auroc"],
        width,
        label="AUROC",
    )

    bars2 = ax.bar(
        x + width / 2,
        plot_df["pr_auc"],
        width,
        label="PR-AUC",
    )

    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_xlabel("Method")
    ax.set_ylim(0, 1)

    ax.set_xticks(x)
    ax.set_xticklabels(
        plot_df["method"],
        rotation=25,
        ha="right",
    )

    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            value = bar.get_height()

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_ml_accuracy_chart(
    ml_2025,
):
    output_path = (
        FIG_ML_ACCURACY
        / "ml_primary_accuracy_2025.png"
    )

    save_bar_chart(
        ml_2025,
        metric="accuracy",
        title=(
            "2025 ML Primary Accuracy "
            "(5-Fold Stratified OOF)"
        ),
        ylabel="Accuracy",
        output_path=output_path,
    )


def save_temporal_change_chart(
    master,
    metric,
    title,
    output_path,
):
    ml = master[
        master["method_group"]
        == "Machine Learning"
    ].copy()

    pivot = ml.pivot(
        index="method",
        columns="dataset",
        values=metric,
    )

    pivot = pivot.reindex(
        [
            "Logistic Regression",
            "Random Forest",
            "Gradient Boosting",
        ]
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    x = np.arange(
        len(pivot)
    )

    width = 0.36

    bars1 = ax.bar(
        x - width / 2,
        pivot["2025"],
        width,
        label="2025 OOF",
    )

    bars2 = ax.bar(
        x + width / 2,
        pivot["2026_Q1"],
        width,
        label="2026 Q1 temporal",
    )

    ax.set_title(title)
    ax.set_ylabel(metric.upper().replace("_", "-"))
    ax.set_xlabel("ML model")
    ax.set_ylim(0, 1)

    ax.set_xticks(x)
    ax.set_xticklabels(
        pivot.index,
        rotation=20,
        ha="right",
    )

    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            value = bar.get_height()

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def create_figures(master):
    header("CREATING COMPARISON FIGURES")

    data_2025 = master[
        master["dataset"] == "2025"
    ].copy()

    data_2026 = master[
        master["dataset"] == "2026_Q1"
    ].copy()

    ml_2025 = data_2025[
        data_2025["method_group"]
        == "Machine Learning"
    ].copy()

    # --------------------------------------------------------
    # 2025 figures
    # --------------------------------------------------------

    save_bar_chart(
        data_2025,
        "auroc",
        "2025 AUROC: ML vs Traditional Methods",
        "AUROC",
        FIG_2025_AUROC
        / "2025_auroc_comparison.png",
        baseline=0.5,
    )

    save_bar_chart(
        data_2025,
        "pr_auc",
        "2025 PR-AUC: ML vs Traditional Methods",
        "PR-AUC",
        FIG_2025_PRAUC
        / "2025_pr_auc_comparison.png",
    )

    save_grouped_ranking_chart(
        data_2025,
        "2025 Ranking Performance: ML vs Traditional Methods",
        FIG_2025_COMBINED
        / "2025_auroc_pr_auc_combined.png",
    )

    # --------------------------------------------------------
    # 2026 Q1 figures
    # --------------------------------------------------------

    save_bar_chart(
        data_2026,
        "auroc",
        "2026 Q1 AUROC: ML vs Traditional Methods",
        "AUROC",
        FIG_2026_AUROC
        / "2026_Q1_auroc_comparison.png",
        baseline=0.5,
    )

    save_bar_chart(
        data_2026,
        "pr_auc",
        "2026 Q1 PR-AUC: ML vs Traditional Methods",
        "PR-AUC",
        FIG_2026_PRAUC
        / "2026_Q1_pr_auc_comparison.png",
    )

    save_grouped_ranking_chart(
        data_2026,
        "2026 Q1 Ranking Performance: ML vs Traditional Methods",
        FIG_2026_COMBINED
        / "2026_Q1_auroc_pr_auc_combined.png",
    )

    # --------------------------------------------------------
    # ML-only primary accuracy
    # --------------------------------------------------------

    save_ml_accuracy_chart(
        ml_2025
    )

    # --------------------------------------------------------
    # Temporal ML comparisons
    # --------------------------------------------------------

    save_temporal_change_chart(
        master,
        "auroc",
        "ML AUROC: 2025 Development vs 2026 Q1 Temporal Evaluation",
        FIG_ML_TEMPORAL
        / "ml_auroc_temporal_comparison.png",
    )

    save_temporal_change_chart(
        master,
        "pr_auc",
        "ML PR-AUC: 2025 Development vs 2026 Q1 Temporal Evaluation",
        FIG_ML_TEMPORAL
        / "ml_pr_auc_temporal_comparison.png",
    )

    save_temporal_change_chart(
        master,
        "f1",
        "ML F1: 2025 Development vs 2026 Q1 Temporal Evaluation",
        FIG_ML_TEMPORAL
        / "ml_f1_temporal_comparison.png",
    )

    print(
        "\nFigures saved in separate folders under:"
        f"\n{FIGURE_ROOT}"
    )


def build_summary(master):
    header("BUILDING FINAL SUMMARY")

    common = master[
        [
            "dataset",
            "method",
            "method_group",
            "auroc",
            "pr_auc",
        ]
    ].copy()

    summary_rows = []

    for dataset in [
        "2025",
        "2026_Q1",
    ]:
        subset = common[
            common["dataset"] == dataset
        ]

        best_auroc = subset.loc[
            subset["auroc"].idxmax()
        ]

        best_pr = subset.loc[
            subset["pr_auc"].idxmax()
        ]

        summary_rows.append(
            {
                "dataset": dataset,
                "comparison": "Best AUROC",
                "method": str(
                    best_auroc["method"]
                ),
                "score": float(
                    best_auroc["auroc"]
                ),
            }
        )

        summary_rows.append(
            {
                "dataset": dataset,
                "comparison": "Best PR-AUC",
                "method": str(
                    best_pr["method"]
                ),
                "score": float(
                    best_pr["pr_auc"]
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    path = (
        TABLE_DIR
        / "best_method_summary.csv"
    )

    summary.to_csv(
        path,
        index=False,
    )

    print("\nBest common ranking results:")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    return summary


def main():
    header(
        "PHARMAAI - FINAL ML VS TRADITIONAL COMPARISON"
    )

    print(
        "\nCommon comparison metrics:"
        "\n  AUROC"
        "\n  PR-AUC"
    )

    print(
        "\nML-only metrics:"
        "\n  Primary 2025 OOF accuracy"
        "\n  Precision"
        "\n  Recall"
        "\n  F1"
    )

    print(
        "\nTraditional methods are NOT assigned an "
        "artificial accuracy."
    )

    traditional = load_traditional_results()

    master = build_master_table(
        traditional
    )

    build_common_metric_table(
        master
    )

    build_ml_only_table(
        master
    )

    create_figures(
        master
    )

    build_summary(
        master
    )

    header("FINAL COMPARISON COMPLETE")

    print(
        "\nOutput structure:"
        f"\n{COMPARISON_ROOT}"
        "\n"
        "\n  tables/"
        "\n    final_method_comparison.csv"
        "\n    common_ranking_metrics.csv"
        "\n    ml_detailed_metrics.csv"
        "\n    best_method_summary.csv"
        "\n"
        "\n  figures/"
        "\n    01_2025_auroc/"
        "\n    02_2025_pr_auc/"
        "\n    03_2025_combined/"
        "\n    04_2026_Q1_auroc/"
        "\n    05_2026_Q1_pr_auc/"
        "\n    06_2026_Q1_combined/"
        "\n    07_ml_primary_accuracy/"
        "\n    08_ml_temporal_change/"
    )

    print(
        "\nNOTE:"
        "\nAUROC and PR-AUC are the direct common comparison "
        "between ML and traditional methods."
        "\nAccuracy is shown only for the three ML models "
        "using the agreed 2025 OOF primary accuracy."
    )


if __name__ == "__main__":
    main()
