from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PharmaAI - Aligned Top-K Ranking Evaluation
# ============================================================
#
# Fixes the control-key mismatch discovered in the first Top-K run.
#
# ML key format:
#   DRUG NAME || STANDARDIZED OUTCOME
#
# Traditional key format:
#   exposureId__outcomeId
#
# Both are mapped back to ONE canonical OMOP ID key:
#   exposureId__outcomeId
#
# The script REFUSES to calculate Top-K unless all six methods
# contain exactly the same canonical OMOP controls for a period.
#
# Metrics:
#   Hits@K
#   Precision@K
#   Recall@K
#
# K:
#   10, 20, 50
# ============================================================


PROJECT_ROOT = Path(r"C:\Users\mamid\OneDrive\Desktop\PharmaAI")

RESULTS_ROOT = PROJECT_ROOT / "results"

OMOP_FILE = (
    PROJECT_ROOT / "data" / "raw" / "reference"
    / "omop_reference_set.csv"
)

K_VALUES = [10, 20, 50]

ML_FILES = {
    "Logistic Regression": {
        "2025": RESULTS_ROOT / "ml" / "logistic_regression"
        / "logistic_regression_predictions_2025_oof.csv",
        "2026_Q1": RESULTS_ROOT / "ml" / "logistic_regression"
        / "logistic_regression_predictions_2026_Q1.csv",
    },
    "Random Forest": {
        "2025": RESULTS_ROOT / "ml" / "random_forest"
        / "random_forest_predictions_2025_oof.csv",
        "2026_Q1": RESULTS_ROOT / "ml" / "random_forest"
        / "random_forest_predictions_2026_Q1.csv",
    },
    "Gradient Boosting": {
        "2025": RESULTS_ROOT / "ml" / "gradient_boosting"
        / "gradient_boosting_predictions_2025_oof.csv",
        "2026_Q1": RESULTS_ROOT / "ml" / "gradient_boosting"
        / "gradient_boosting_predictions_2026_Q1.csv",
    },
}

TRADITIONAL_FILES = {
    "2025": RESULTS_ROOT / "traditional_full" / "evaluation"
    / "traditional_control_scores_2025.csv",
    "2026_Q1": RESULTS_ROOT / "traditional_full" / "evaluation"
    / "traditional_control_scores_2026_Q1.csv",
}

OUTPUT_ROOT = RESULTS_ROOT / "top_k_ranking_aligned"
TABLE_DIR = OUTPUT_ROOT / "tables"
RANKED_DIR = OUTPUT_ROOT / "ranked_controls"
DIAGNOSTIC_DIR = OUTPUT_ROOT / "diagnostics"

for d in [TABLE_DIR, RANKED_DIR, DIAGNOSTIC_DIR]:
    d.mkdir(parents=True, exist_ok=True)


OUTCOME_TRANSLATION = {
    "OMOP ACUTE LIVER FAILURE 1": "ACUTE LIVER INJURY",
    "OMOP ACUTE RENAL FAILURE 1": "ACUTE KIDNEY INJURY",
    "OMOP ACUTE MYOCARDIAL INFARCTION 1": "ACUTE MYOCARDIAL INFARCTION",
    "HOI UPPER GI 3": "GI BLEED",
}


def header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")


def norm_text(x):
    """
    Normalize names exactly enough for deterministic OMOP-to-ML matching.

    Example:
        "Sodium Phosphate, Monobasic"
        -> "SODIUM PHOSPHATE MONOBASIC"

    This is NOT fuzzy matching. It only standardizes case, punctuation,
    and whitespace so the OMOP drug name is represented the same way
    as the ML reference-control key.
    """
    import re

    s = str(x).strip().upper()

    # Replace punctuation with spaces while keeping letters/numbers.
    s = re.sub(r"[^A-Z0-9]+", " ", s)

    # Collapse repeated whitespace.
    s = " ".join(s.split())

    return s


def clean_id(x):
    if pd.isna(x):
        return None

    s = str(x).strip()

    if s.endswith(".0"):
        s = s[:-2]

    return s


def find_column(df, candidates, description):
    lookup = {str(c).lower(): c for c in df.columns}

    for c in candidates:
        if c in df.columns:
            return c

        if c.lower() in lookup:
            return lookup[c.lower()]

    raise ValueError(
        f"Could not find {description}.\n"
        f"Tried: {candidates}\n"
        f"Available: {list(df.columns)}"
    )


def load_omop_crosswalk():
    header("BUILDING CANONICAL OMOP CONTROL CROSSWALK")

    require_file(OMOP_FILE)
    omop = pd.read_csv(OMOP_FILE)

    required = [
        "exposureId",
        "exposureName",
        "outcomeId",
        "outcomeName",
        "groundTruth",
    ]

    missing = [c for c in required if c not in omop.columns]

    if missing:
        raise ValueError(f"OMOP file missing columns: {missing}")

    omop["exposure_id_clean"] = omop["exposureId"].map(clean_id)
    omop["outcome_id_clean"] = omop["outcomeId"].map(clean_id)

    omop["canonical_control_key"] = (
        omop["exposure_id_clean"]
        + "__"
        + omop["outcome_id_clean"]
    )

    omop["drug_norm"] = omop["exposureName"].map(norm_text)

    source_outcome = omop["outcomeName"].map(norm_text)

    unexpected = sorted(
        set(source_outcome) - set(OUTCOME_TRANSLATION)
    )

    if unexpected:
        raise ValueError(
            "Unexpected OMOP outcomes found:\n"
            + "\n".join(unexpected)
        )

    omop["standardized_outcome"] = (
        source_outcome.map(OUTCOME_TRANSLATION).map(norm_text)
    )

    omop["ml_control_key_norm"] = (
        omop["drug_norm"]
        + " || "
        + omop["standardized_outcome"]
    )

    if omop["canonical_control_key"].duplicated().any():
        raise ValueError(
            "Duplicate canonical OMOP ID keys found."
        )

    if omop["ml_control_key_norm"].duplicated().any():
        dup = omop.loc[
            omop["ml_control_key_norm"].duplicated(False),
            [
                "exposureName",
                "outcomeName",
                "ml_control_key_norm",
            ],
        ]
        dup.to_csv(
            DIAGNOSTIC_DIR / "duplicate_ml_crosswalk_keys.csv",
            index=False,
        )
        raise ValueError(
            "The OMOP name-based ML crosswalk is not unique. "
            "See duplicate_ml_crosswalk_keys.csv."
        )

    crosswalk = omop[
        [
            "canonical_control_key",
            "ml_control_key_norm",
            "exposureId",
            "exposureName",
            "outcomeId",
            "outcomeName",
            "standardized_outcome",
            "groundTruth",
        ]
    ].copy()

    crosswalk.to_csv(
        DIAGNOSTIC_DIR / "omop_control_crosswalk.csv",
        index=False,
    )

    print(f"OMOP controls: {len(crosswalk):,}")
    print(
        f"Positive controls: "
        f"{int((crosswalk['groundTruth'] == 1).sum()):,}"
    )
    print(
        f"Negative controls: "
        f"{int((crosswalk['groundTruth'] == 0).sum()):,}"
    )

    return crosswalk


def load_ml(path, method, dataset, crosswalk):
    require_file(path)
    df = pd.read_csv(path)

    key_col = find_column(
        df,
        ["reference_control_key", "control_key"],
        "ML control key",
    )

    label_col = find_column(
        df,
        ["groundTruth", "ground_truth", "label", "y_true"],
        "ground truth",
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
        "ML probability/score",
    )

    def normalize_ml_control_key(value):
        raw = str(value)

        if "||" not in raw:
            raise ValueError(
                f"{method} {dataset}: unexpected ML control key format: {raw}"
            )

        drug, outcome = raw.split("||", 1)

        return (
            norm_text(drug)
            + " || "
            + norm_text(outcome)
        )

    temp = pd.DataFrame({
        "ml_control_key_norm": df[key_col].map(normalize_ml_control_key),
        "file_groundTruth": pd.to_numeric(
            df[label_col], errors="raise"
        ).astype(int),
        "score": pd.to_numeric(df[score_col], errors="coerce"),
    })

    merged = temp.merge(
        crosswalk,
        on="ml_control_key_norm",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    unmatched = merged[merged["_merge"] != "both"].copy()

    if len(unmatched):
        unmatched.to_csv(
            DIAGNOSTIC_DIR
            / f"{dataset}_{method.lower().replace(' ', '_')}_unmatched.csv",
            index=False,
        )
        raise ValueError(
            f"{method} {dataset}: {len(unmatched)} ML controls "
            "could not be mapped back to OMOP IDs."
        )

    label_mismatch = merged[
        merged["file_groundTruth"] != merged["groundTruth"]
    ]

    if len(label_mismatch):
        raise ValueError(
            f"{method} {dataset}: ground-truth mismatch after OMOP mapping."
        )

    out = merged[
        [
            "canonical_control_key",
            "exposureName",
            "outcomeName",
            "standardized_outcome",
            "groundTruth",
            "score",
        ]
    ].copy()

    out["method"] = method
    out["dataset"] = dataset

    return out


def load_traditional(path, dataset, crosswalk):
    require_file(path)
    df = pd.read_csv(path)

    key_col = find_column(
        df,
        ["reference_control_key", "control_key"],
        "traditional control key",
    )

    label_col = find_column(
        df,
        ["groundTruth", "ground_truth"],
        "ground truth",
    )

    score_cols = {
        "ROR": find_column(df, ["ror_score", "ror"], "ROR score"),
        "PRR": find_column(df, ["prr_score", "prr"], "PRR score"),
        "IC": find_column(df, ["ic_score", "ic"], "IC score"),
    }

    canonical_lookup = crosswalk[
        [
            "canonical_control_key",
            "exposureName",
            "outcomeName",
            "standardized_outcome",
            "groundTruth",
        ]
    ].copy()

    outputs = {}

    for method, score_col in score_cols.items():
        temp = pd.DataFrame({
            "canonical_control_key": df[key_col].map(clean_id),
            "file_groundTruth": pd.to_numeric(
                df[label_col], errors="raise"
            ).astype(int),
            "score": pd.to_numeric(df[score_col], errors="coerce"),
        })

        merged = temp.merge(
            canonical_lookup,
            on="canonical_control_key",
            how="left",
            validate="many_to_one",
            indicator=True,
        )

        unmatched = merged[merged["_merge"] != "both"].copy()

        if len(unmatched):
            unmatched.to_csv(
                DIAGNOSTIC_DIR
                / f"{dataset}_{method.lower()}_unmatched.csv",
                index=False,
            )
            raise ValueError(
                f"{method} {dataset}: {len(unmatched)} traditional "
                "controls could not be mapped to the OMOP reference set."
            )

        mismatch = merged[
            merged["file_groundTruth"] != merged["groundTruth"]
        ]

        if len(mismatch):
            raise ValueError(
                f"{method} {dataset}: ground-truth mismatch."
            )

        out = merged[
            [
                "canonical_control_key",
                "exposureName",
                "outcomeName",
                "standardized_outcome",
                "groundTruth",
                "score",
            ]
        ].copy()

        out["method"] = method
        out["dataset"] = dataset

        outputs[method] = out

    return outputs


def require_identical_universe(method_frames, dataset):
    header(f"STRICT CONTROL-UNIVERSE CHECK: {dataset}")

    methods = list(method_frames)
    reference_method = methods[0]

    reference = method_frames[reference_method]

    reference_keys = set(reference["canonical_control_key"])

    if reference["canonical_control_key"].duplicated().any():
        raise ValueError(
            f"{reference_method} {dataset} has duplicate canonical controls."
        )

    reference_labels = dict(
        zip(
            reference["canonical_control_key"],
            reference["groundTruth"],
        )
    )

    print(
        f"Reference: {reference_method} "
        f"({len(reference_keys)} controls)"
    )

    for method in methods[1:]:
        df = method_frames[method]

        if df["canonical_control_key"].duplicated().any():
            raise ValueError(
                f"{method} {dataset} has duplicate canonical controls."
            )

        keys = set(df["canonical_control_key"])

        same = keys == reference_keys

        print(
            f"{reference_method} vs {method}: "
            f"{'SAME' if same else 'DIFFERENT'}"
        )

        if not same:
            left_only = sorted(reference_keys - keys)
            right_only = sorted(keys - reference_keys)

            pd.DataFrame(
                {"canonical_control_key": left_only}
            ).to_csv(
                DIAGNOSTIC_DIR
                / f"{dataset}_{reference_method.lower().replace(' ', '_')}_only_vs_{method.lower().replace(' ', '_')}.csv",
                index=False,
            )

            pd.DataFrame(
                {"canonical_control_key": right_only}
            ).to_csv(
                DIAGNOSTIC_DIR
                / f"{dataset}_{method.lower().replace(' ', '_')}_only_vs_{reference_method.lower().replace(' ', '_')}.csv",
                index=False,
            )

            raise ValueError(
                f"STOPPED: {method} does not use the exact same "
                f"OMOP controls as {reference_method} for {dataset}."
            )

        labels = dict(
            zip(df["canonical_control_key"], df["groundTruth"])
        )

        disagreement = [
            key
            for key in reference_keys
            if labels[key] != reference_labels[key]
        ]

        if disagreement:
            raise ValueError(
                f"STOPPED: {method} has {len(disagreement)} "
                f"ground-truth label disagreements in {dataset}."
            )

    print(
        "\nPASS: all six methods use exactly the same "
        "canonical OMOP controls and labels."
    )


def evaluate_method(df, method, dataset):
    if df["score"].isna().any():
        missing = int(df["score"].isna().sum())
        raise ValueError(
            f"STOPPED: {method} {dataset} has {missing} missing scores. "
            "A strict same-universe Top-K comparison requires one score "
            "for every control."
        )

    ranked = df.sort_values(
        ["score", "canonical_control_key"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    ranked["rank"] = np.arange(1, len(ranked) + 1)

    total_positives = int(ranked["groundTruth"].sum())
    total_controls = len(ranked)

    rows = []

    for k in K_VALUES:
        if k > total_controls:
            raise ValueError(
                f"K={k} exceeds number of controls ({total_controls})."
            )

        top = ranked.head(k)
        hits = int(top["groundTruth"].sum())

        rows.append({
            "dataset": dataset,
            "method": method,
            "k": k,
            "hits_at_k": hits,
            "precision_at_k": hits / k,
            "recall_at_k": hits / total_positives,
            "total_controls": total_controls,
            "total_positives": total_positives,
            "total_negatives": total_controls - total_positives,
        })

    return pd.DataFrame(rows), ranked


def main():
    header("PHARMAAI - ALIGNED TOP-K RANKING EVALUATION")

    print("\nCanonical comparison key:")
    print("  exposureId__outcomeId")

    print("\nK values:")
    print(f"  {K_VALUES}")

    crosswalk = load_omop_crosswalk()

    all_results = []

    for dataset in ["2025", "2026_Q1"]:
        header(f"LOADING AND ALIGNING: {dataset}")

        method_frames = {}

        for method, files in ML_FILES.items():
            frame = load_ml(
                files[dataset],
                method,
                dataset,
                crosswalk,
            )
            method_frames[method] = frame
            print(f"{method}: {len(frame):,} mapped controls")

        traditional = load_traditional(
            TRADITIONAL_FILES[dataset],
            dataset,
            crosswalk,
        )

        for method, frame in traditional.items():
            method_frames[method] = frame
            print(f"{method}: {len(frame):,} mapped controls")

        require_identical_universe(
            method_frames,
            dataset,
        )

        header(f"FINAL ALIGNED TOP-K RESULTS: {dataset}")

        dataset_rank_dir = RANKED_DIR / dataset
        dataset_rank_dir.mkdir(parents=True, exist_ok=True)

        for method, frame in method_frames.items():
            result, ranked = evaluate_method(
                frame,
                method,
                dataset,
            )

            all_results.append(result)

            safe_method = method.lower().replace(" ", "_")

            ranked.to_csv(
                dataset_rank_dir
                / f"{safe_method}_ranked_controls.csv",
                index=False,
            )

            print(f"\n{method}")

            for _, row in result.iterrows():
                print(
                    f"  K={int(row['k']):>2}: "
                    f"Hits={int(row['hits_at_k']):>2}, "
                    f"Precision@K={row['precision_at_k']:.4f}, "
                    f"Recall@K={row['recall_at_k']:.4f}"
                )

    final = pd.concat(all_results, ignore_index=True)

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
        ["dataset", "k", "method"]
    ).reset_index(drop=True)

    final.to_csv(
        TABLE_DIR / "aligned_top_k_ranking_results.csv",
        index=False,
    )

    for dataset in ["2025", "2026_Q1"]:
        subset = final[final["dataset"] == dataset].copy()

        subset.to_csv(
            TABLE_DIR / f"aligned_top_k_ranking_{dataset}.csv",
            index=False,
        )

        precision = subset.pivot(
            index="method",
            columns="k",
            values="precision_at_k",
        ).reset_index()

        precision.columns = [
            "method" if c == "method" else f"precision_at_{int(c)}"
            for c in precision.columns
        ]

        recall = subset.pivot(
            index="method",
            columns="k",
            values="recall_at_k",
        ).reset_index()

        recall.columns = [
            "method" if c == "method" else f"recall_at_{int(c)}"
            for c in recall.columns
        ]

        compact = precision.merge(recall, on="method")

        compact.to_csv(
            TABLE_DIR / f"aligned_top_k_compact_{dataset}.csv",
            index=False,
        )

    header("ALIGNED TOP-K EVALUATION COMPLETE")

    print(
        "\nPASS means the reported Top-K results are based on "
        "the exact same canonical OMOP controls for all six methods."
    )

    print(f"\nResults saved to:\n{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
