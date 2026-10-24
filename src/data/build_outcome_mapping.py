from pathlib import Path
import re
import sys
import unicodedata

import pandas as pd


# ============================================================
# PharmaAI - Build OMOP Outcome Mapping Candidates
# ============================================================
#
# PURPOSE
# -------
# The OMOP reference set contains 4 broad outcomes, while FAERS
# adverse events use MedDRA terminology. Exact string matching
# therefore produced zero complete drug-event matches.
#
# This script:
#   1. Reads the existing FAERS signal tables.
#   2. Collects the actual adverse-event terms present in FAERS.
#   3. Creates CONSERVATIVE candidate MedDRA-term mappings for
#      each of the 4 OMOP outcomes.
#   4. Saves the candidates for human review.
#
# IMPORTANT
# ---------
# This script DOES NOT create final ground-truth labels.
# Candidate mappings must be reviewed before they are used.
# It does not modify existing signal tables.
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\mamid\OneDrive\Desktop\PharmaAI")

SIGNAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "signals"
)

REFERENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reference"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "reference"
)

REFERENCE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DATASETS = {
    "2025": (
        SIGNAL_DIR
        / "drug_event_signals_2025.parquet"
    ),
    "2026_Q1": (
        SIGNAL_DIR
        / "drug_event_signals_2026_Q1.parquet"
    ),
}


# ============================================================
# 2. OMOP OUTCOME SEARCH DEFINITIONS
# ============================================================
#
# These terms are used ONLY to find candidate FAERS MedDRA terms
# for review. They are NOT automatically accepted as equivalent
# clinical concepts.
#
# The OMOP reference set has four outcomes:
# - Acute Liver Injury
# - Acute Kidney Injury
# - Acute Myocardial Infarction
# - GI Bleed
#
# ============================================================

OUTCOME_SEARCH_TERMS = {
    "Acute Liver Injury": [
        "LIVER",
        "HEPATIC",
        "HEPATOCELLULAR",
        "HEPATITIS",
    ],

    "Acute Kidney Injury": [
        "KIDNEY",
        "RENAL",
        "NEPHR",
    ],

    "Acute Myocardial Infarction": [
        "MYOCARDIAL INFARCTION",
        "MYOCARDIAL",
        "CARDIOGENIC SHOCK",
    ],

    "GI Bleed": [
        "GASTROINTESTINAL HAEMORRHAGE",
        "GASTROINTESTINAL HEMORRHAGE",
        "GI HAEMORRHAGE",
        "GI HEMORRHAGE",
        "HAEMATEMESIS",
        "HEMATEMESIS",
        "MELAENA",
        "MELENA",
        "PEPTIC ULCER HAEMORRHAGE",
        "PEPTIC ULCER HEMORRHAGE",
    ],
}


# ============================================================
# 3. TEXT NORMALIZATION
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = unicodedata.normalize(
        "NFKD",
        str(value).strip()
    )

    value = "".join(
        character
        for character in value
        if not unicodedata.combining(character)
    )

    value = value.upper()

    value = re.sub(
        r"[^A-Z0-9]+",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


# ============================================================
# 4. LOAD FAERS EVENT VOCABULARY
# ============================================================

def load_faers_events():

    print("\n" + "=" * 80)
    print("LOADING FAERS ADVERSE-EVENT VOCABULARY")
    print("=" * 80)

    frames = []

    for dataset_name, file_path in DATASETS.items():

        if not file_path.exists():
            raise FileNotFoundError(
                f"Signal table not found:\n{file_path}"
            )

        print(
            f"\nReading adverse events from "
            f"{dataset_name}..."
        )

        df = pd.read_parquet(
            file_path,
            columns=[
                "adverse_event",
                "pair_report_count",
            ],
        )

        df = df.dropna(
            subset=["adverse_event"]
        ).copy()

        df["dataset"] = dataset_name

        frames.append(df)

    combined = pd.concat(
        frames,
        ignore_index=True
    )

    combined[
        "normalized_event"
    ] = combined[
        "adverse_event"
    ].map(
        normalize_text
    )

    # Aggregate support across all drug-event pairs so that
    # frequently observed FAERS event terms can be prioritized
    # during manual review.
    event_summary = (
        combined
        .groupby(
            [
                "adverse_event",
                "normalized_event"
            ],
            as_index=False
        )
        .agg(
            total_pair_report_count=(
                "pair_report_count",
                "sum"
            ),
            datasets_present=(
                "dataset",
                "nunique"
            ),
        )
    )

    event_summary = (
        event_summary
        .sort_values(
            "total_pair_report_count",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print(
        f"\nUnique FAERS adverse-event terms: "
        f"{len(event_summary):,}"
    )

    return event_summary


# ============================================================
# 5. FIND CANDIDATE TERMS
# ============================================================

def find_candidates(
    event_summary
):

    print("\n" + "=" * 80)
    print("SEARCHING FOR OUTCOME-MAPPING CANDIDATES")
    print("=" * 80)

    candidate_frames = []

    for (
        omop_outcome,
        search_terms
    ) in OUTCOME_SEARCH_TERMS.items():

        print(
            f"\nOMOP outcome: {omop_outcome}"
        )

        outcome_candidates = []

        for search_term in search_terms:

            normalized_search = normalize_text(
                search_term
            )

            mask = (
                event_summary[
                    "normalized_event"
                ]
                .str.contains(
                    re.escape(
                        normalized_search
                    ),
                    regex=True,
                    na=False
                )
            )

            found = (
                event_summary[
                    mask
                ]
                .copy()
            )

            if found.empty:
                continue

            found[
                "omop_outcome"
            ] = omop_outcome

            found[
                "matched_search_term"
            ] = search_term

            outcome_candidates.append(
                found
            )

        if outcome_candidates:

            outcome_df = pd.concat(
                outcome_candidates,
                ignore_index=True
            )

            outcome_df = (
                outcome_df
                .sort_values(
                    "total_pair_report_count",
                    ascending=False
                )
                .drop_duplicates(
                    subset=[
                        "omop_outcome",
                        "adverse_event",
                    ]
                )
                .reset_index(drop=True)
            )

            print(
                f"Candidate FAERS terms found: "
                f"{len(outcome_df):,}"
            )

            candidate_frames.append(
                outcome_df
            )

        else:

            print(
                "Candidate FAERS terms found: 0"
            )

    if not candidate_frames:

        return pd.DataFrame()

    candidates = pd.concat(
        candidate_frames,
        ignore_index=True
    )

    # These fields are deliberately left for human review.
    candidates[
        "review_decision"
    ] = ""

    candidates[
        "review_notes"
    ] = ""

    candidates[
        "mapping_status"
    ] = "REVIEW_REQUIRED"

    return candidates


# ============================================================
# 6. SAVE REVIEW FILES
# ============================================================

def save_outputs(
    event_summary,
    candidates
):

    vocabulary_file = (
        RESULTS_DIR
        / "faers_adverse_event_vocabulary.csv"
    )

    event_summary.to_csv(
        vocabulary_file,
        index=False
    )

    print(
        f"\nFAERS event vocabulary saved:\n"
        f"{vocabulary_file}"
    )

    candidate_file = (
        REFERENCE_DIR
        / "omop_faers_outcome_mapping_candidates.csv"
    )

    candidates.to_csv(
        candidate_file,
        index=False
    )

    print(
        f"\nOutcome-mapping candidates saved:\n"
        f"{candidate_file}"
    )

    if not candidates.empty:

        summary = (
            candidates
            .groupby(
                "omop_outcome",
                as_index=False
            )
            .agg(
                candidate_terms=(
                    "adverse_event",
                    "nunique"
                )
            )
        )

    else:

        summary = pd.DataFrame(
            columns=[
                "omop_outcome",
                "candidate_terms",
            ]
        )

    summary_file = (
        RESULTS_DIR
        / "outcome_mapping_candidate_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print(
        f"\nCandidate summary saved:\n"
        f"{summary_file}"
    )

    return candidate_file


# ============================================================
# 7. MAIN
# ============================================================

def main():

    print("\n" + "=" * 80)
    print(
        "PHARMAAI - BUILD OUTCOME "
        "MAPPING CANDIDATES"
    )
    print("=" * 80)

    print(
        "\nThis is a terminology-mapping "
        "diagnostic step."
    )

    print(
        "It does NOT create final "
        "ground-truth labels."
    )

    event_summary = (
        load_faers_events()
    )

    candidates = (
        find_candidates(
            event_summary
        )
    )

    candidate_file = (
        save_outputs(
            event_summary,
            candidates
        )
    )

    print("\n" + "=" * 80)
    print("OUTCOME-MAPPING CANDIDATES COMPLETE")
    print("=" * 80)

    print(
        f"\nTotal candidate rows: "
        f"{len(candidates):,}"
    )

    print(
        "\nNEXT STEP:"
    )

    print(
        "Review the candidate mapping CSV "
        "before any term is accepted as an "
        "OMOP-to-FAERS outcome mapping."
    )

    print(
        "\nReview file:\n"
        f"{candidate_file}"
    )

    print(
        "\nDo NOT use REVIEW_REQUIRED rows "
        "as ground truth yet."
    )


# ============================================================
# 8. RUN
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\nOutcome mapping interrupted."
        )

        sys.exit(1)

    except Exception as error:

        print(
            "\nOutcome mapping failed:"
        )

        print(error)

        raise
