from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# PharmaAI - Final Results Assembly
# ============================================================
#
# Creates:
# 1. Final ML performance comparison
# 2. ML vs Traditional AUROC comparison
# 3. ML vs Traditional PR-AUC comparison
# 4. Precision@K comparison
# 5. Recall@K comparison
# 6. Final CSV tables
# 7. Final conclusions text file
#
# IMPORTANT:
# - Uses validated results only.
# - 2025 ML metrics are OOF results.
# - 2026 Q1 is temporal evaluation.
# - Traditional methods are compared using ranking metrics.
# - Traditional accuracy is NOT invented.
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\mamid\OneDrive\Desktop\PharmaAI"
)

RESULTS_ROOT = PROJECT_ROOT / "results"

OUTPUT_ROOT = RESULTS_ROOT / "final_results"

TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"

ML_FIG_DIR = FIGURE_DIR / "01_ml_models"
RANKING_FIG_DIR = FIGURE_DIR / "02_ml_vs_traditional"
TOPK_FIG_DIR = FIGURE_DIR / "03_top_k"

for folder in [
    TABLE_DIR,
    ML_FIG_DIR,
    RANKING_FIG_DIR,
    TOPK_FIG_DIR,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 1. VALIDATED ML RESULTS
# ============================================================

ml_2025 = pd.DataFrame({
    "Method": [
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting",
    ],
    "Accuracy": [
        0.6913,
        0.7785,
        0.7718,
    ],
    "Precision": [
        0.8310,
        0.8191,
        0.8315,
    ],
    "Recall": [
        0.6344,
        0.8280,
        0.7957,
    ],
    "F1": [
        0.7195,
        0.8235,
        0.8132,
    ],
    "AUROC": [
        0.8024,
        0.8337,
        0.8110,
    ],
    "PR_AUC": [
        0.8882,
        0.9028,
        0.8888,
    ],
})


ml_2026 = pd.DataFrame({
    "Method": [
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting",
    ],
    "Precision": [
        0.9375,
        0.8235,
        0.8125,
    ],
    "Recall": [
        0.2000,
        0.5600,
        0.8667,
    ],
    "F1": [
        0.3297,
        0.6667,
        0.8387,
    ],
    "AUROC": [
        0.6452,
        0.5990,
        0.7438,
    ],
    "PR_AUC": [
        0.8519,
        0.8121,
        0.8846,
    ],
})


# ============================================================
# 2. VALIDATED TRADITIONAL RESULTS
# ============================================================

traditional_2025 = pd.DataFrame({
    "Method": [
        "ROR",
        "PRR",
        "IC",
    ],
    "AUROC": [
        0.5346,
        0.5344,
        0.5328,
    ],
    "PR_AUC": [
        0.6252,
        0.6246,
        0.6227,
    ],
})


traditional_2026 = pd.DataFrame({
    "Method": [
        "ROR",
        "PRR",
        "IC",
    ],
    "AUROC": [
        0.4910,
        0.4986,
        0.4936,
    ],
    "PR_AUC": [
        0.6884,
        0.7014,
        0.6944,
    ],
})


# ============================================================
# 3. VALIDATED TOP-K RESULTS
# ============================================================

topk_rows = [

    # ---------------- 2025 ----------------

    ["2025", "Logistic Regression", 10, 10, 1.0000, 0.1075],
    ["2025", "Logistic Regression", 20, 20, 1.0000, 0.2151],
    ["2025", "Logistic Regression", 50, 48, 0.9600, 0.5161],

    ["2025", "Random Forest", 10, 10, 1.0000, 0.1075],
    ["2025", "Random Forest", 20, 20, 1.0000, 0.2151],
    ["2025", "Random Forest", 50, 48, 0.9600, 0.5161],

    ["2025", "Gradient Boosting", 10, 10, 1.0000, 0.1075],
    ["2025", "Gradient Boosting", 20, 20, 1.0000, 0.2151],
    ["2025", "Gradient Boosting", 50, 46, 0.9200, 0.4946],

    ["2025", "ROR", 10, 6, 0.6000, 0.0645],
    ["2025", "ROR", 20, 11, 0.5500, 0.1183],
    ["2025", "ROR", 50, 32, 0.6400, 0.3441],

    ["2025", "PRR", 10, 6, 0.6000, 0.0645],
    ["2025", "PRR", 20, 11, 0.5500, 0.1183],
    ["2025", "PRR", 50, 32, 0.6400, 0.3441],

    ["2025", "IC", 10, 5, 0.5000, 0.0538],
    ["2025", "IC", 20, 11, 0.5500, 0.1183],
    ["2025", "IC", 50, 32, 0.6400, 0.3441],

    # ---------------- 2026 Q1 ----------------

    ["2026_Q1", "Logistic Regression", 10, 10, 1.0000, 0.1333],
    ["2026_Q1", "Logistic Regression", 20, 18, 0.9000, 0.2400],
    ["2026_Q1", "Logistic Regression", 50, 41, 0.8200, 0.5467],

    ["2026_Q1", "Random Forest", 10, 8, 0.8000, 0.1067],
    ["2026_Q1", "Random Forest", 20, 16, 0.8000, 0.2133],
    ["2026_Q1", "Random Forest", 50, 41, 0.8200, 0.5467],

    ["2026_Q1", "Gradient Boosting", 10, 10, 1.0000, 0.1333],
    ["2026_Q1", "Gradient Boosting", 20, 19, 0.9500, 0.2533],
    ["2026_Q1", "Gradient Boosting", 50, 42, 0.8400, 0.5600],

    ["2026_Q1", "ROR", 10, 4, 0.4000, 0.0533],
    ["2026_Q1", "ROR", 20, 12, 0.6000, 0.1600],
    ["2026_Q1", "ROR", 50, 38, 0.7600, 0.5067],

    ["2026_Q1", "PRR", 10, 5, 0.5000, 0.0667],
    ["2026_Q1", "PRR", 20, 12, 0.6000, 0.1600],
    ["2026_Q1", "PRR", 50, 38, 0.7600, 0.5067],

    ["2026_Q1", "IC", 10, 5, 0.5000, 0.0667],
    ["2026_Q1", "IC", 20, 12, 0.6000, 0.1600],
    ["2026_Q1", "IC", 50, 38, 0.7600, 0.5067],
]


topk = pd.DataFrame(
    topk_rows,
    columns=[
        "Dataset",
        "Method",
        "K",
        "Hits",
        "Precision_at_K",
        "Recall_at_K",
    ],
)


# ============================================================
# 4. SAVE FINAL TABLES
# ============================================================

ml_2025.to_csv(
    TABLE_DIR / "ml_results_2025.csv",
    index=False
)

ml_2026.to_csv(
    TABLE_DIR / "ml_results_2026_Q1.csv",
    index=False
)

traditional_2025.to_csv(
    TABLE_DIR / "traditional_results_2025.csv",
    index=False
)

traditional_2026.to_csv(
    TABLE_DIR / "traditional_results_2026_Q1.csv",
    index=False
)

topk.to_csv(
    TABLE_DIR / "final_top_k_results.csv",
    index=False
)


# ============================================================
# 5. FIGURE 1
# ML PRIMARY ACCURACY - 2025
# ============================================================

plt.figure(figsize=(9, 6))

bars = plt.bar(
    ml_2025["Method"],
    ml_2025["Accuracy"]
)

plt.title(
    "Machine-Learning Model Accuracy – 2025 OOF Evaluation"
)

plt.ylabel("Accuracy")
plt.ylim(0, 1)

plt.xticks(
    rotation=15,
    ha="right"
)

for bar, value in zip(
    bars,
    ml_2025["Accuracy"]
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value * 100:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    ML_FIG_DIR / "01_ml_primary_accuracy_2025.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 6. FIGURE 2
# ML 2025 PERFORMANCE
# ============================================================

metrics = [
    "Precision",
    "Recall",
    "F1",
    "AUROC",
    "PR_AUC",
]

x = np.arange(
    len(ml_2025["Method"])
)

width = 0.15

plt.figure(figsize=(12, 7))

for i, metric in enumerate(metrics):

    plt.bar(
        x + (i - 2) * width,
        ml_2025[metric],
        width,
        label=metric.replace("_", "-")
    )

plt.xticks(
    x,
    ml_2025["Method"]
)

plt.ylabel("Score")

plt.ylim(0, 1)

plt.title(
    "Machine-Learning Performance – 2025 OOF Evaluation"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    ML_FIG_DIR / "02_ml_performance_2025.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 7. FIGURE 3
# ML 2026 TEMPORAL PERFORMANCE
# ============================================================

metrics_2026 = [
    "Precision",
    "Recall",
    "F1",
    "AUROC",
    "PR_AUC",
]

x = np.arange(
    len(ml_2026["Method"])
)

plt.figure(figsize=(12, 7))

for i, metric in enumerate(metrics_2026):

    plt.bar(
        x + (i - 2) * width,
        ml_2026[metric],
        width,
        label=metric.replace("_", "-")
    )

plt.xticks(
    x,
    ml_2026["Method"]
)

plt.ylabel("Score")
plt.ylim(0, 1)

plt.title(
    "Machine-Learning Performance – 2026 Q1 Temporal Evaluation"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    ML_FIG_DIR / "03_ml_performance_2026_Q1.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. ML VS TRADITIONAL - AUROC 2025
# ============================================================

comparison_2025 = pd.concat(
    [
        ml_2025[
            [
                "Method",
                "AUROC",
                "PR_AUC",
            ]
        ],
        traditional_2025,
    ],
    ignore_index=True
)


plt.figure(figsize=(10, 6))

bars = plt.bar(
    comparison_2025["Method"],
    comparison_2025["AUROC"]
)

plt.title(
    "ML vs Traditional Methods – AUROC (2025)"
)

plt.ylabel("AUROC")

plt.ylim(0, 1)

plt.xticks(
    rotation=20,
    ha="right"
)

for bar, value in zip(
    bars,
    comparison_2025["AUROC"]
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.015,
        f"{value:.3f}",
        ha="center",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    RANKING_FIG_DIR
    / "01_ml_vs_traditional_auroc_2025.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 9. ML VS TRADITIONAL - PR-AUC 2025
# ============================================================

plt.figure(figsize=(10, 6))

bars = plt.bar(
    comparison_2025["Method"],
    comparison_2025["PR_AUC"]
)

plt.title(
    "ML vs Traditional Methods – PR-AUC (2025)"
)

plt.ylabel("PR-AUC")

plt.ylim(0, 1)

plt.xticks(
    rotation=20,
    ha="right"
)

for bar, value in zip(
    bars,
    comparison_2025["PR_AUC"]
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.015,
        f"{value:.3f}",
        ha="center",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    RANKING_FIG_DIR
    / "02_ml_vs_traditional_pr_auc_2025.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. ML VS TRADITIONAL - 2026 Q1
# ============================================================

comparison_2026 = pd.concat(
    [
        ml_2026[
            [
                "Method",
                "AUROC",
                "PR_AUC",
            ]
        ],
        traditional_2026,
    ],
    ignore_index=True
)


plt.figure(figsize=(10, 6))

bars = plt.bar(
    comparison_2026["Method"],
    comparison_2026["AUROC"]
)

plt.title(
    "ML vs Traditional Methods – AUROC (2026 Q1)"
)

plt.ylabel("AUROC")

plt.ylim(0, 1)

plt.xticks(
    rotation=20,
    ha="right"
)

for bar, value in zip(
    bars,
    comparison_2026["AUROC"]
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.015,
        f"{value:.3f}",
        ha="center",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    RANKING_FIG_DIR
    / "03_ml_vs_traditional_auroc_2026_Q1.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


plt.figure(figsize=(10, 6))

bars = plt.bar(
    comparison_2026["Method"],
    comparison_2026["PR_AUC"]
)

plt.title(
    "ML vs Traditional Methods – PR-AUC (2026 Q1)"
)

plt.ylabel("PR-AUC")

plt.ylim(0, 1)

plt.xticks(
    rotation=20,
    ha="right"
)

for bar, value in zip(
    bars,
    comparison_2026["PR_AUC"]
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.015,
        f"{value:.3f}",
        ha="center",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    RANKING_FIG_DIR
    / "04_ml_vs_traditional_pr_auc_2026_Q1.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. TOP-K PLOTTING FUNCTION
# ============================================================

method_order = [
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
    "ROR",
    "PRR",
    "IC",
]


def plot_topk(
    dataset,
    metric,
    ylabel,
    title,
    filename
):

    subset = topk[
        topk["Dataset"] == dataset
    ]

    plt.figure(
        figsize=(11, 7)
    )

    for method in method_order:

        method_data = subset[
            subset["Method"] == method
        ].sort_values("K")

        plt.plot(
            method_data["K"],
            method_data[metric],
            marker="o",
            linewidth=2,
            label=method
        )

    plt.xlabel("K")

    plt.ylabel(ylabel)

    plt.xticks(
        [10, 20, 50]
    )

    plt.ylim(
        0,
        1.05
    )

    plt.title(title)

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        TOPK_FIG_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 12. TOP-K FIGURES
# ============================================================

plot_topk(
    "2025",
    "Precision_at_K",
    "Precision@K",
    "Top-K Precision – 2025",
    "01_precision_at_k_2025.png"
)

plot_topk(
    "2025",
    "Recall_at_K",
    "Recall@K",
    "Top-K Recall – 2025",
    "02_recall_at_k_2025.png"
)

plot_topk(
    "2026_Q1",
    "Precision_at_K",
    "Precision@K",
    "Top-K Precision – 2026 Q1",
    "03_precision_at_k_2026_Q1.png"
)

plot_topk(
    "2026_Q1",
    "Recall_at_K",
    "Recall@K",
    "Top-K Recall – 2026 Q1",
    "04_recall_at_k_2026_Q1.png"
)


# ============================================================
# 13. FINAL RESULTS SUMMARY TABLE
# ============================================================

summary = pd.DataFrame({

    "Finding": [

        "Best 2025 ML accuracy",

        "Best 2025 AUROC",

        "Best 2025 PR-AUC",

        "Best traditional 2025 AUROC",

        "Best traditional 2025 PR-AUC",

        "Best 2026 Q1 ML AUROC",

        "Best 2026 Q1 ML PR-AUC",

        "Best 2026 Q1 F1",

        "Best 2025 Precision@10",

        "Best 2026 Q1 Precision@10",
    ],

    "Result": [

        "Random Forest – 77.85%",

        "Random Forest – 0.8337",

        "Random Forest – 0.9028",

        "ROR – 0.5346",

        "ROR – 0.6252",

        "Gradient Boosting – 0.7438",

        "Gradient Boosting – 0.8846",

        "Gradient Boosting – 0.8387",

        "LR / RF / GB – 1.00",

        "LR / GB – 1.00",
    ],
})


summary.to_csv(
    TABLE_DIR / "final_results_summary.csv",
    index=False
)


# ============================================================
# 14. FINAL CONCLUSIONS
# ============================================================

conclusions = """
PHARMAAI - FINAL RESULTS AND CONCLUSIONS
========================================

EVALUATION DESIGN
-----------------
The evaluation used an external OMOP reference set rather than
ROR, PRR, or IC-derived labels. This avoids using the traditional
methods themselves to define the machine-learning ground truth.

The 2025 period was used for model development. Machine-learning
performance for 2025 was evaluated using stratified 5-fold
out-of-fold predictions.

The 2026 Q1 period was used as a later temporal evaluation period.

The final aligned comparison contained:
- 149 OMOP controls in 2025
- 103 OMOP controls in 2026 Q1

All six methods were verified to use the same canonical OMOP
control universe for the final Top-K comparison.


RQ1
---
To what extent do machine-learning models improve identification
and prioritization of potential drug-adverse-event safety signals
compared with traditional disproportionality methods?

RESULT:

Machine-learning methods showed substantially stronger ranking
performance than ROR, PRR, and IC within the evaluated
OMOP-matched FAERS reference set.

In 2025:

Random Forest achieved:
- AUROC = 0.8337
- PR-AUC = 0.9028

The strongest traditional result was ROR:
- AUROC = 0.5346
- PR-AUC = 0.6252

Therefore, compared with the strongest traditional method,
Random Forest improved:

- AUROC by approximately 0.299
- PR-AUC by approximately 0.278

These results indicate stronger discrimination and prioritization
of reference-positive versus reference-negative controls by the
machine-learning models in the evaluated reference set.


RQ2
---
How do Logistic Regression, Random Forest, and Gradient Boosting
compare with ROR, PRR, and IC?

2025 MACHINE-LEARNING RESULTS
-----------------------------

Logistic Regression:
- Accuracy = 69.13%
- Precision = 83.10%
- Recall = 63.44%
- F1 = 71.95%
- AUROC = 0.8024
- PR-AUC = 0.8882

Random Forest:
- Accuracy = 77.85%
- Precision = 81.91%
- Recall = 82.80%
- F1 = 82.35%
- AUROC = 0.8337
- PR-AUC = 0.9028

Gradient Boosting:
- Accuracy = 77.18%
- Precision = 83.15%
- Recall = 79.57%
- F1 = 81.32%
- AUROC = 0.8110
- PR-AUC = 0.8888

Random Forest produced the strongest overall 2025 development
performance, including the highest accuracy, F1, AUROC, and
PR-AUC among the three machine-learning models.


2026 Q1 TEMPORAL RESULTS
------------------------

Gradient Boosting showed the strongest overall temporal
performance:

- Precision = 81.25%
- Recall = 86.67%
- F1 = 83.87%
- AUROC = 0.7438
- PR-AUC = 0.8846

This suggests that Gradient Boosting maintained stronger
performance in the later evaluation period than the other
machine-learning models.

Logistic Regression had high precision but substantially lower
recall in 2026 Q1, indicating that it identified fewer of the
positive controls at the fixed 0.50 classification threshold.


TOP-K PRIORITIZATION
--------------------

The Top-K evaluation directly measured how effectively each
method prioritized known positive controls near the top of its
ranking.

For 2025:

At K=10:
- Logistic Regression: 10/10 positives
- Random Forest: 10/10 positives
- Gradient Boosting: 10/10 positives
- ROR: 6/10 positives
- PRR: 6/10 positives
- IC: 5/10 positives

At K=20:
- Logistic Regression: 20/20 positives
- Random Forest: 20/20 positives
- Gradient Boosting: 20/20 positives
- ROR: 11/20 positives
- PRR: 11/20 positives
- IC: 11/20 positives

This provides strong evidence that the machine-learning methods
were more effective at early prioritization within the evaluated
2025 control set.


For 2026 Q1:

At K=10:
- Logistic Regression: 10/10 positives
- Random Forest: 8/10 positives
- Gradient Boosting: 10/10 positives
- ROR: 4/10 positives
- PRR: 5/10 positives
- IC: 5/10 positives

At K=20:
- Logistic Regression: 18/20 positives
- Random Forest: 16/20 positives
- Gradient Boosting: 19/20 positives
- ROR: 12/20 positives
- PRR: 12/20 positives
- IC: 12/20 positives

Gradient Boosting therefore demonstrated particularly strong
prioritization in the later temporal period.


OVERALL CONCLUSION
------------------

Within the evaluated OMOP-matched FAERS reference set, the
machine-learning approaches provided stronger discrimination and
early prioritization of reference-positive drug-outcome controls
than the traditional ROR, PRR, and IC methods.

Random Forest showed the strongest overall development-period
performance in 2025, while Gradient Boosting showed the strongest
overall performance in the 2026 Q1 temporal evaluation.

The results support the use of machine learning as a
prioritization aid for pharmacovigilance signal review rather
than as a replacement for clinical or regulatory assessment.

The identified drug-event associations should therefore be
interpreted as potential safety-signal candidates requiring
further review, not as evidence that a drug caused an adverse
event.


IMPORTANT LIMITATIONS
---------------------

1. The external reference evaluation contains a relatively small
   number of matched OMOP controls.

2. The 2026 Q1 evaluation is temporal, but many controls overlap
   with those present in 2025. It should not be described as a
   completely unseen-control validation set.

3. The 2025 period covers a full year whereas 2026 Q1 covers one
   quarter, which can affect count-derived features.

4. The OMOP-to-FAERS outcome mapping is a narrow project mapping
   and should not be described as an officially validated MedDRA
   SMQ mapping.

5. FAERS is a spontaneous-reporting database. Disproportionality
   patterns and model predictions do not establish causality.

6. These findings therefore support prioritization performance
   within the evaluated reference set and should not be
   generalized as proof that machine learning universally
   outperforms traditional pharmacovigilance methods.
"""


conclusion_path = (
    OUTPUT_ROOT
    / "FINAL_RESULTS_AND_CONCLUSIONS.txt"
)

conclusion_path.write_text(
    conclusions,
    encoding="utf-8"
)


# ============================================================
# FINISHED
# ============================================================

print("=" * 70)
print("PHARMAAI FINAL RESULTS ASSEMBLY COMPLETE")
print("=" * 70)

print(
    f"\nTables saved to:\n{TABLE_DIR}"
)

print(
    f"\nFigures saved to:\n{FIGURE_DIR}"
)

print(
    f"\nConclusions saved to:\n{conclusion_path}"
)

print("\nCreated figures:")

print(
    "1. ML primary accuracy"
)

print(
    "2. ML 2025 performance"
)

print(
    "3. ML 2026 Q1 temporal performance"
)

print(
    "4. 2025 AUROC: ML vs traditional"
)

print(
    "5. 2025 PR-AUC: ML vs traditional"
)

print(
    "6. 2026 Q1 AUROC: ML vs traditional"
)

print(
    "7. 2026 Q1 PR-AUC: ML vs traditional"
)

print(
    "8. Precision@K - 2025"
)

print(
    "9. Recall@K - 2025"
)

print(
    "10. Precision@K - 2026 Q1"
)

print(
    "11. Recall@K - 2026 Q1"
)