from pathlib import Path
import re
import sys
import unicodedata
import numpy as np
import pandas as pd

# ============================================================
# PharmaAI - Build External Reference Labels
# ============================================================
# Matches the OHDSI OMOP reference set to existing PharmaAI
# FAERS drug-event signal tables.
#
# IMPORTANT:
# - Does NOT recalculate ROR, PRR, or IC.
# - Does NOT modify existing signal parquet files.
# - OMOP groundTruth: 1 = positive control, 0 = negative control.
# - Unmatched FAERS pairs are NOT labeled as negative.
# - First pass uses conservative normalized exact-name matching.
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\mamid\OneDrive\Desktop\PharmaAI")

REFERENCE_INPUT = (
    PROJECT_ROOT / "data" / "raw" / "reference" / "omop_reference_set.csv"
)
SIGNAL_DIR = PROJECT_ROOT / "data" / "processed" / "signals"
REFERENCE_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "reference"
RESULTS_DIR = PROJECT_ROOT / "results" / "reference"

REFERENCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "2025": SIGNAL_DIR / "drug_event_signals_2025.parquet",
    "2026_Q1": SIGNAL_DIR / "drug_event_signals_2026_Q1.parquet",
}


def normalize_text(value):
    """Normalize text without guessing synonyms."""
    if pd.isna(value):
        return ""

    value = unicodedata.normalize("NFKD", str(value).strip())
    value = "".join(
        c for c in value if not unicodedata.combining(c)
    )
    value = value.upper()
    value = re.sub(r"[^A-Z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def load_reference_set():
    print("\n" + "=" * 80)
    print("LOADING OMOP REFERENCE SET")
    print("=" * 80)

    if not REFERENCE_INPUT.exists():
        raise FileNotFoundError(
            f"OMOP reference CSV not found:\n{REFERENCE_INPUT}\n\n"
            "Place omop_reference_set.csv in data/raw/reference/"
        )

    reference = pd.read_csv(REFERENCE_INPUT)

    required = ["exposureName", "outcomeName", "groundTruth"]
    missing = [c for c in required if c not in reference.columns]
    if missing:
        raise ValueError(
            "Missing required OMOP columns: " + ", ".join(missing)
        )

    reference["groundTruth"] = pd.to_numeric(
        reference["groundTruth"], errors="coerce"
    )
    if not reference["groundTruth"].isin([0, 1]).all():
        raise ValueError("groundTruth must contain only 0 and 1.")

    reference["groundTruth"] = reference["groundTruth"].astype("int8")
    reference["normalized_drug"] = reference["exposureName"].map(normalize_text)
    reference["normalized_outcome"] = reference["outcomeName"].map(normalize_text)
    reference["reference_pair_key"] = (
        reference["normalized_drug"] + "||" + reference["normalized_outcome"]
    )

    # Detect conflicting labels after normalization.
    conflicts = (
        reference.groupby("reference_pair_key")["groundTruth"]
        .nunique()
    )
    if (conflicts > 1).any():
        raise ValueError(
            "Conflicting OMOP labels found after name normalization."
        )

    reference = reference.drop_duplicates(
        subset=["reference_pair_key"]
    ).reset_index(drop=True)

    print(f"Reference pairs: {len(reference):,}")
    print(f"Positive controls: {(reference['groundTruth'] == 1).sum():,}")
    print(f"Negative controls: {(reference['groundTruth'] == 0).sum():,}")
    print(f"Unique reference drugs: {reference['normalized_drug'].nunique():,}")
    print(f"Unique reference outcomes: {reference['normalized_outcome'].nunique():,}")

    return reference


def load_signal_table(dataset_name, file_path):
    print("\n" + "-" * 80)
    print(f"LOADING FAERS SIGNAL TABLE - {dataset_name}")
    print("-" * 80)

    if not file_path.exists():
        raise FileNotFoundError(f"Signal table not found:\n{file_path}")

    signals = pd.read_parquet(
        file_path,
        columns=["suspect_drug", "adverse_event"],
    )

    signals = signals.drop_duplicates(
        subset=["suspect_drug", "adverse_event"]
    ).reset_index(drop=True)

    signals["normalized_drug"] = signals["suspect_drug"].map(normalize_text)
    signals["normalized_event"] = signals["adverse_event"].map(normalize_text)
    signals["faers_pair_key"] = (
        signals["normalized_drug"] + "||" + signals["normalized_event"]
    )

    print(f"Unique FAERS drug-event pairs: {len(signals):,}")
    return signals


def exact_pair_match(reference, signals, dataset_name):
    keep = [
        "exposureName",
        "outcomeName",
        "groundTruth",
        "normalized_drug",
        "normalized_outcome",
        "reference_pair_key",
    ]

    for col in [
        "exposureId",
        "outcomeId",
        "indicationId",
        "indicationName",
        "comparatorId",
        "comparatorName",
        "comparatorType",
    ]:
        if col in reference.columns:
            keep.append(col)

    matched = signals.merge(
        reference[keep],
        left_on=["normalized_drug", "normalized_event"],
        right_on=["normalized_drug", "normalized_outcome"],
        how="inner",
        validate="many_to_one",
    )

    matched["dataset"] = dataset_name
    matched["reference_source"] = "OHDSI OMOP Reference Set"
    matched["match_method"] = "exact_normalized_name"
    matched["reference_label"] = matched["groundTruth"].astype("int8")

    return matched


def create_reports(reference, signals, matched, dataset_name):
    faers_drugs = set(signals["normalized_drug"].unique())

    drug_coverage = (
        reference[["exposureName", "normalized_drug"]]
        .drop_duplicates()
        .copy()
    )
    drug_coverage["drug_name_found_in_faers"] = (
        drug_coverage["normalized_drug"].isin(faers_drugs)
    )

    matched_keys = set(matched["reference_pair_key"].unique())
    matched_mask = reference["reference_pair_key"].isin(matched_keys)

    pos = reference["groundTruth"] == 1
    neg = reference["groundTruth"] == 0

    summary = pd.DataFrame([{
        "dataset": dataset_name,
        "faers_unique_pairs": len(signals),
        "reference_pairs": len(reference),
        "reference_positive_controls": int(pos.sum()),
        "reference_negative_controls": int(neg.sum()),
        "reference_unique_drugs": len(drug_coverage),
        "reference_drugs_found_in_faers": int(
            drug_coverage["drug_name_found_in_faers"].sum()
        ),
        "exact_pair_matches": int(matched_mask.sum()),
        "matched_positive_controls": int((matched_mask & pos).sum()),
        "matched_negative_controls": int((matched_mask & neg).sum()),
        "overall_match_rate": (
            float(matched_mask.sum() / len(reference))
            if len(reference) else np.nan
        ),
    }])

    unmatched = reference.loc[~matched_mask].copy()
    return summary, drug_coverage, unmatched


def save_outputs(matched, summary, drug_coverage, unmatched, dataset_name):
    label_columns = [
        "dataset",
        "suspect_drug",
        "adverse_event",
        "exposureName",
        "outcomeName",
        "reference_label",
        "reference_source",
        "match_method",
    ]

    for col in ["exposureId", "outcomeId", "indicationName", "comparatorName"]:
        if col in matched.columns:
            label_columns.append(col)

    labels = (
        matched[label_columns]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    csv_path = (
        REFERENCE_OUTPUT_DIR
        / f"pharmaai_reference_labels_{dataset_name}.csv"
    )
    parquet_path = (
        REFERENCE_OUTPUT_DIR
        / f"pharmaai_reference_labels_{dataset_name}.parquet"
    )
    unmatched_path = (
        RESULTS_DIR
        / f"unmatched_omop_controls_{dataset_name}.csv"
    )
    drug_path = (
        RESULTS_DIR
        / f"omop_drug_name_coverage_{dataset_name}.csv"
    )

    labels.to_csv(csv_path, index=False)
    labels.to_parquet(parquet_path, index=False)
    unmatched.to_csv(unmatched_path, index=False)
    drug_coverage.to_csv(drug_path, index=False)

    print(f"\nMatched labels: {len(labels):,}")
    print(f"Unmatched OMOP controls: {len(unmatched):,}")
    print(f"Saved labels:\n{csv_path}\n{parquet_path}")
    print(f"Saved unmatched controls:\n{unmatched_path}")
    print(f"Saved drug-name coverage:\n{drug_path}")

    return labels


def process_dataset(reference, dataset_name, file_path):
    signals = load_signal_table(dataset_name, file_path)
    matched = exact_pair_match(reference, signals, dataset_name)

    summary, drug_coverage, unmatched = create_reports(
        reference, signals, matched, dataset_name
    )

    save_outputs(
        matched,
        summary,
        drug_coverage,
        unmatched,
        dataset_name,
    )

    print("\nMATCH SUMMARY")
    print(summary.to_string(index=False))
    return summary


def main():
    print("\n" + "=" * 80)
    print("PHARMAAI - BUILD EXTERNAL REFERENCE LABELS")
    print("=" * 80)
    print("This script does NOT create labels from ROR, PRR, or IC.")

    reference = load_reference_set()

    summaries = []
    for dataset_name, file_path in DATASETS.items():
        summaries.append(
            process_dataset(reference, dataset_name, file_path)
        )

    combined = pd.concat(summaries, ignore_index=True)
    summary_path = RESULTS_DIR / "reference_match_summary.csv"
    combined.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("REFERENCE LABEL BUILD COMPLETE")
    print("=" * 80)
    print(f"Summary saved:\n{summary_path}")
    print(
        "\nNEXT: Inspect match coverage before adding synonym, "
        "ingredient, or adverse-event concept mapping."
    )
    print("Unmatched FAERS pairs must NOT be treated as negatives.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nReference-label build interrupted.")
        sys.exit(1)
    except Exception as error:
        print("\nReference-label build failed:")
        print(error)
        raise

#python build_reference_labels.py 2>&1 | Tee-Object -FilePath "../../results/logs/07_build_reference_labels.txt"