from pathlib import Path
import re
import sys
import unicodedata

import pandas as pd


# ============================================================
# PharmaAI - Build Mapped OMOP Reference Labels
# ============================================================
#
# PURPOSE
# -------
# The first reference-label experiment required an exact match:
#
#     OMOP outcome name == FAERS adverse-event name
#
# That produced zero complete matches because the OMOP reference
# set uses broad outcome concepts while FAERS uses MedDRA event
# terminology.
#
# This script uses the separately reviewed/approved mapping file:
#
#     omop_faers_outcome_mapping_approved.csv
#
# to expand each OMOP drug-outcome control into approved FAERS
# drug-event pairs, and then matches those pairs against the
# EXISTING 2025 and 2026 Q1 signal tables.
#
# IMPORTANT
# ---------
# - Does NOT recalculate ROR, PRR, or IC.
# - Does NOT modify the original signal tables.
# - Does NOT modify build_reference_labels.py.
# - Does NOT label unmatched FAERS pairs as negatives.
# - Only OMOP controls actually observed as mapped FAERS pairs
#   become labeled rows.
#
# OMOP groundTruth is preserved:
#     1 = positive control
#     0 = negative control
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\mamid\OneDrive\Desktop\PharmaAI"
)

SIGNAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "signals"
)

REFERENCE_RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "reference"
)

REFERENCE_PROCESSED_DIR = (
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

REFERENCE_PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. INPUT FILES
# ============================================================

OMOP_FILE = (
    REFERENCE_RAW_DIR
    / "omop_reference_set.csv"
)

MAPPING_FILE = (
    REFERENCE_PROCESSED_DIR
    / "omop_faers_outcome_mapping_approved.csv"
)

SIGNAL_FILES = {
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
# 3. OMOP OUTCOME NAME ALIGNMENT
# ============================================================
#
# The OMOP CSV uses internal/source outcome names that differ
# from the standardized names used in our approved FAERS mapping.
# We explicitly translate only these four known values.
# No fuzzy outcome-name matching is used.
# ============================================================


# ============================================================
# 4. TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    Conservative normalization used for matching names.

    It:
    - removes leading/trailing spaces
    - removes accents
    - converts to uppercase
    - replaces punctuation with spaces
    - collapses repeated spaces

    It does NOT perform fuzzy matching, stemming, synonym
    expansion, or automatic medical concept guessing.
    """

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
# 5. EXPLICIT OUTCOME TRANSLATION
# ============================================================

OMOP_OUTCOME_NAME_MAP = {
    normalize_text("OMOP Acute Liver Failure 1"):
        "Acute Liver Injury",

    normalize_text("OMOP Acute Renal Failure 1"):
        "Acute Kidney Injury",

    normalize_text("OMOP Acute myocardial Infarction  1"):
        "Acute Myocardial Infarction",

    normalize_text("HOI Upper GI #3"):
        "GI Bleed",
}


def standardize_omop_outcome(value):
    """Translate only the four known OMOP outcome names."""

    normalized = normalize_text(value)

    return OMOP_OUTCOME_NAME_MAP.get(
        normalized
    )


# ============================================================
# 6. COLUMN HELPERS
# ============================================================

def find_column(
    df,
    candidates,
    file_description
):
    """
    Find a column even if capitalization differs slightly.
    """

    lookup = {
        str(column).lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lookup:

            return lookup[
                candidate.lower()
            ]

    raise ValueError(
        f"\nCould not find the required column in "
        f"{file_description}.\n"
        f"Expected one of: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# 5. LOAD OMOP REFERENCE SET
# ============================================================

def load_omop_reference():

    print("\n" + "=" * 80)
    print("LOADING OMOP REFERENCE SET")
    print("=" * 80)

    if not OMOP_FILE.exists():

        raise FileNotFoundError(
            f"\nOMOP reference CSV not found:\n"
            f"{OMOP_FILE}\n\n"
            "Expected location:\n"
            "data/raw/reference/"
            "omop_reference_set.csv"
        )

    omop = pd.read_csv(
        OMOP_FILE
    )

    drug_col = find_column(
        omop,
        [
            "exposureName",
            "exposure_name",
        ],
        "OMOP reference set"
    )

    outcome_col = find_column(
        omop,
        [
            "outcomeName",
            "outcome_name",
        ],
        "OMOP reference set"
    )

    truth_col = find_column(
        omop,
        [
            "groundTruth",
            "ground_truth",
        ],
        "OMOP reference set"
    )

    exposure_id_col = None

    try:

        exposure_id_col = find_column(
            omop,
            [
                "exposureId",
                "exposure_id",
            ],
            "OMOP reference set"
        )

    except ValueError:

        pass

    outcome_id_col = None

    try:

        outcome_id_col = find_column(
            omop,
            [
                "outcomeId",
                "outcome_id",
            ],
            "OMOP reference set"
        )

    except ValueError:

        pass

    keep = pd.DataFrame()

    if exposure_id_col is not None:
        keep[
            "omop_exposure_id"
        ] = omop[
            exposure_id_col
        ]

    keep[
        "omop_drug"
    ] = omop[
        drug_col
    ]

    if outcome_id_col is not None:
        keep[
            "omop_outcome_id"
        ] = omop[
            outcome_id_col
        ]

    keep[
        "omop_outcome"
    ] = omop[
        outcome_col
    ]

    keep[
        "groundTruth"
    ] = pd.to_numeric(
        omop[
            truth_col
        ],
        errors="coerce"
    )

    if keep[
        "groundTruth"
    ].isna().any():

        raise ValueError(
            "OMOP groundTruth contains "
            "missing/non-numeric values."
        )

    keep[
        "groundTruth"
    ] = keep[
        "groundTruth"
    ].astype(int)

    invalid_truth = (
        ~keep[
            "groundTruth"
        ].isin(
            [0, 1]
        )
    )

    if invalid_truth.any():

        raise ValueError(
            "OMOP groundTruth must contain "
            "only 0 and 1."
        )

    keep[
        "normalized_drug"
    ] = keep[
        "omop_drug"
    ].map(
        normalize_text
    )

    # Preserve the source outcome name for auditability.
    keep[
        "omop_outcome_source"
    ] = keep[
        "omop_outcome"
    ]

    # Translate the four known OMOP source names to the
    # standardized outcome names used by the approved mapping.
    keep[
        "standardized_outcome"
    ] = keep[
        "omop_outcome_source"
    ].map(
        standardize_omop_outcome
    )

    unknown_outcomes = (
        keep.loc[
            keep[
                "standardized_outcome"
            ].isna(),
            "omop_outcome_source"
        ]
        .drop_duplicates()
        .tolist()
    )

    if unknown_outcomes:

        raise ValueError(
            "\nUnexpected OMOP outcome name(s). "
            "No fuzzy mapping was attempted:\n"
            + "\n".join(
                f"  - {value}"
                for value in unknown_outcomes
            )
        )

    keep[
        "omop_outcome"
    ] = keep[
        "standardized_outcome"
    ]

    keep[
        "normalized_outcome"
    ] = keep[
        "standardized_outcome"
    ].map(
        normalize_text
    )

    print(
        f"\nReference controls: "
        f"{len(keep):,}"
    )

    print(
        "Positive controls: "
        f"{(keep['groundTruth'] == 1).sum():,}"
    )

    print(
        "Negative controls: "
        f"{(keep['groundTruth'] == 0).sum():,}"
    )

    print(
        "Unique reference drugs: "
        f"{keep['normalized_drug'].nunique():,}"
    )

    print(
        "Unique reference outcomes: "
        f"{keep['normalized_outcome'].nunique():,}"
    )

    print(
        "\nOMOP outcome-name alignment:"
    )

    alignment = (
        keep[
            [
                "omop_outcome_source",
                "standardized_outcome",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "standardized_outcome"
        )
    )

    for _, row in alignment.iterrows():

        print(
            f"  {row['omop_outcome_source']} "
            f"-> {row['standardized_outcome']}"
        )

    return keep


# ============================================================
# 6. LOAD APPROVED OUTCOME MAPPING
# ============================================================

def load_mapping():

    print("\n" + "=" * 80)
    print("LOADING APPROVED OUTCOME MAPPING")
    print("=" * 80)

    if not MAPPING_FILE.exists():

        raise FileNotFoundError(
            f"\nApproved mapping file not found:\n"
            f"{MAPPING_FILE}\n\n"
            "Expected location:\n"
            "data/processed/reference/"
            "omop_faers_outcome_mapping_approved.csv"
        )

    mapping = pd.read_csv(
        MAPPING_FILE
    )

    outcome_col = find_column(
        mapping,
        [
            "omop_outcome",
            "outcomeName",
            "outcome_name",
        ],
        "approved mapping file"
    )

    event_col = find_column(
        mapping,
        [
            "faers_adverse_event",
            "adverse_event",
        ],
        "approved mapping file"
    )

    mapping = mapping.copy()

    mapping[
        "omop_outcome"
    ] = mapping[
        outcome_col
    ]

    mapping[
        "faers_adverse_event"
    ] = mapping[
        event_col
    ]

    mapping[
        "normalized_outcome"
    ] = mapping[
        "omop_outcome"
    ].map(
        normalize_text
    )

    mapping[
        "normalized_event"
    ] = mapping[
        "faers_adverse_event"
    ].map(
        normalize_text
    )

    mapping = (
        mapping
        .drop_duplicates(
            subset=[
                "normalized_outcome",
                "normalized_event",
            ]
        )
        .reset_index(drop=True)
    )

    if mapping[
        "normalized_outcome"
    ].eq("").any():

        raise ValueError(
            "Approved mapping contains "
            "a blank OMOP outcome."
        )

    if mapping[
        "normalized_event"
    ].eq("").any():

        raise ValueError(
            "Approved mapping contains "
            "a blank FAERS adverse event."
        )

    print(
        f"\nApproved mapping rows: "
        f"{len(mapping):,}"
    )

    print(
        "Mapped OMOP outcomes: "
        f"{mapping['normalized_outcome'].nunique():,}"
    )

    print("\nApproved terms per outcome:")

    counts = (
        mapping
        .groupby(
            "omop_outcome"
        )[
            "normalized_event"
        ]
        .nunique()
        .sort_index()
    )

    for outcome, count in counts.items():

        print(
            f"  {outcome}: {count:,}"
        )

    return mapping[
        [
            "omop_outcome",
            "faers_adverse_event",
            "normalized_outcome",
            "normalized_event",
        ]
    ]


# ============================================================
# 7. EXPAND OMOP CONTROLS THROUGH MAPPING
# ============================================================

def expand_reference_controls(
    omop,
    mapping
):

    print("\n" + "=" * 80)
    print("EXPANDING OMOP CONTROLS THROUGH APPROVED MAPPING")
    print("=" * 80)

    expanded = omop.merge(
        mapping[
            [
                "normalized_outcome",
                "faers_adverse_event",
                "normalized_event",
            ]
        ],
        on="normalized_outcome",
        how="left",
        validate="many_to_many",
    )

    missing_mapping = (
        expanded[
            "normalized_event"
        ].isna()
    )

    if missing_mapping.any():

        missing_outcomes = (
            expanded.loc[
                missing_mapping,
                "omop_outcome"
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "\nSome OMOP outcomes do not have "
            "approved FAERS mappings:\n"
            + "\n".join(
                f"  - {x}"
                for x in missing_outcomes
            )
        )

    expanded[
        "reference_control_key"
    ] = (
        expanded[
            "normalized_drug"
        ]
        + " || "
        + expanded[
            "normalized_outcome"
        ]
    )

    expanded[
        "mapped_pair_key"
    ] = (
        expanded[
            "normalized_drug"
        ]
        + " || "
        + expanded[
            "normalized_event"
        ]
    )

    print(
        f"\nOriginal OMOP controls: "
        f"{len(omop):,}"
    )

    print(
        "Expanded drug-event rows: "
        f"{len(expanded):,}"
    )

    return expanded


# ============================================================
# 8. LOAD ONE FAERS SIGNAL TABLE
# ============================================================

def load_signal_table(
    dataset_name,
    file_path
):

    print("\n" + "-" * 80)
    print(
        f"LOADING FAERS SIGNAL TABLE: "
        f"{dataset_name}"
    )
    print("-" * 80)

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nSignal table not found:\n"
            f"{file_path}"
        )

    signal = pd.read_parquet(
        file_path
    )

    required_columns = [
        "suspect_drug",
        "adverse_event",
    ]

    missing = [
        column
        for column in required_columns
        if column not in signal.columns
    ]

    if missing:

        raise ValueError(
            f"\nMissing required signal columns: "
            f"{missing}"
        )

    signal = signal.copy()

    signal[
        "normalized_drug"
    ] = signal[
        "suspect_drug"
    ].map(
        normalize_text
    )

    signal[
        "normalized_event"
    ] = signal[
        "adverse_event"
    ].map(
        normalize_text
    )

    print(
        f"Signal-table rows: "
        f"{len(signal):,}"
    )

    print(
        "Unique normalized drug-event pairs: "
        f"{signal[['normalized_drug', 'normalized_event']].drop_duplicates().shape[0]:,}"
    )

    return signal


# ============================================================
# 9. MATCH REFERENCE CONTROLS TO FAERS
# ============================================================

def match_dataset(
    dataset_name,
    signal,
    expanded
):

    print("\n" + "=" * 80)
    print(
        f"MATCHING MAPPED REFERENCE CONTROLS: "
        f"{dataset_name}"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Drug coverage diagnostic
    # --------------------------------------------------------

    faers_drugs = set(
        signal[
            "normalized_drug"
        ].dropna()
    )

    reference_drugs = set(
        expanded[
            "normalized_drug"
        ].dropna()
    )

    found_drugs = (
        reference_drugs
        & faers_drugs
    )

    print(
        f"\nReference drugs found in FAERS: "
        f"{len(found_drugs):,} of "
        f"{len(reference_drugs):,}"
    )

    # --------------------------------------------------------
    # Event coverage diagnostic
    # --------------------------------------------------------

    faers_events = set(
        signal[
            "normalized_event"
        ].dropna()
    )

    mapped_events = set(
        expanded[
            "normalized_event"
        ].dropna()
    )

    found_events = (
        mapped_events
        & faers_events
    )

    print(
        "Approved mapped events found in FAERS: "
        f"{len(found_events):,} of "
        f"{len(mapped_events):,}"
    )

    # --------------------------------------------------------
    # Exact mapped drug-event merge
    # --------------------------------------------------------

    matched = expanded.merge(
        signal,
        on=[
            "normalized_drug",
            "normalized_event",
        ],
        how="inner",
        suffixes=(
            "_reference",
            "_faers"
        ),
    )

    # Avoid accidental duplicate rows if the signal table ever
    # contains duplicate normalized pairs.
    matched = (
        matched
        .sort_values(
            [
                "reference_control_key",
                "normalized_event",
            ]
        )
        .drop_duplicates(
            subset=[
                "reference_control_key",
                "normalized_event",
            ]
        )
        .reset_index(drop=True)
    )

    matched[
        "dataset"
    ] = dataset_name

    # --------------------------------------------------------
    # Control-level coverage
    #
    # One OMOP drug-outcome control may map to several approved
    # MedDRA event terms. We therefore report:
    #
    # 1. matched mapped drug-event rows
    # 2. unique matched OMOP controls
    #
    # --------------------------------------------------------

    matched_control_keys = set(
        matched[
            "reference_control_key"
        ]
    )

    total_control_keys = set(
        expanded[
            "reference_control_key"
        ]
    )

    matched_controls = (
        expanded[
            expanded[
                "reference_control_key"
            ].isin(
                matched_control_keys
            )
        ]
        [
            [
                "reference_control_key",
                "omop_drug",
                "omop_outcome",
                "groundTruth",
            ]
        ]
        .drop_duplicates(
            subset=[
                "reference_control_key"
            ]
        )
        .reset_index(drop=True)
    )

    matched_positive = int(
        (
            matched_controls[
                "groundTruth"
            ] == 1
        ).sum()
    )

    matched_negative = int(
        (
            matched_controls[
                "groundTruth"
            ] == 0
        ).sum()
    )

    total_positive = int(
        (
            expanded[
                [
                    "reference_control_key",
                    "groundTruth",
                ]
            ]
            .drop_duplicates()[
                "groundTruth"
            ] == 1
        ).sum()
    )

    total_negative = int(
        (
            expanded[
                [
                    "reference_control_key",
                    "groundTruth",
                ]
            ]
            .drop_duplicates()[
                "groundTruth"
            ] == 0
        ).sum()
    )

    print(
        f"\nMatched mapped drug-event rows: "
        f"{len(matched):,}"
    )

    print(
        "Unique matched OMOP controls: "
        f"{len(matched_controls):,} of "
        f"{len(total_control_keys):,}"
    )

    print(
        "Matched positive controls: "
        f"{matched_positive:,} of "
        f"{total_positive:,}"
    )

    print(
        "Matched negative controls: "
        f"{matched_negative:,} of "
        f"{total_negative:,}"
    )

    if total_control_keys:

        coverage = (
            len(matched_controls)
            / len(total_control_keys)
        )

    else:

        coverage = 0.0

    print(
        f"Control coverage: "
        f"{coverage:.2%}"
    )

    # --------------------------------------------------------
    # Unmatched control diagnostics
    # --------------------------------------------------------

    control_table = (
        expanded[
            [
                "reference_control_key",
                "omop_drug",
                "omop_outcome",
                "groundTruth",
            ]
        ]
        .drop_duplicates(
            subset=[
                "reference_control_key"
            ]
        )
        .copy()
    )

    control_table[
        "matched_in_faers"
    ] = control_table[
        "reference_control_key"
    ].isin(
        matched_control_keys
    )

    unmatched_controls = (
        control_table[
            ~control_table[
                "matched_in_faers"
            ]
        ]
        .drop(
            columns=[
                "matched_in_faers"
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Summary row
    # --------------------------------------------------------

    summary = {
        "dataset": dataset_name,
        "omop_controls_total": len(
            total_control_keys
        ),
        "omop_positive_total": (
            total_positive
        ),
        "omop_negative_total": (
            total_negative
        ),
        "reference_drugs_total": len(
            reference_drugs
        ),
        "reference_drugs_found": len(
            found_drugs
        ),
        "approved_events_total": len(
            mapped_events
        ),
        "approved_events_found": len(
            found_events
        ),
        "matched_mapped_pair_rows": len(
            matched
        ),
        "matched_controls_total": len(
            matched_controls
        ),
        "matched_positive_controls": (
            matched_positive
        ),
        "matched_negative_controls": (
            matched_negative
        ),
        "control_coverage": coverage,
    }

    return (
        matched,
        matched_controls,
        unmatched_controls,
        summary,
    )


# ============================================================
# 10. SAVE DATASET OUTPUTS
# ============================================================

def save_dataset_outputs(
    dataset_name,
    matched,
    matched_controls,
    unmatched_controls
):

    # Detailed mapped FAERS rows.
    detailed_csv = (
        REFERENCE_PROCESSED_DIR
        / (
            "pharmaai_mapped_reference_labels_"
            f"{dataset_name}.csv"
        )
    )

    detailed_parquet = (
        REFERENCE_PROCESSED_DIR
        / (
            "pharmaai_mapped_reference_labels_"
            f"{dataset_name}.parquet"
        )
    )

    matched.to_csv(
        detailed_csv,
        index=False
    )

    matched.to_parquet(
        detailed_parquet,
        index=False
    )

    # One row per matched OMOP control.
    control_csv = (
        REFERENCE_PROCESSED_DIR
        / (
            "pharmaai_matched_omop_controls_"
            f"{dataset_name}.csv"
        )
    )

    matched_controls.to_csv(
        control_csv,
        index=False
    )

    # Unmatched controls for diagnostics.
    unmatched_csv = (
        RESULTS_DIR
        / (
            "unmatched_mapped_omop_controls_"
            f"{dataset_name}.csv"
        )
    )

    unmatched_controls.to_csv(
        unmatched_csv,
        index=False
    )

    print(
        "\nSaved mapped labeled rows:\n"
        f"{detailed_csv}"
    )

    print(
        "\nSaved mapped labeled parquet:\n"
        f"{detailed_parquet}"
    )

    print(
        "\nSaved matched OMOP controls:\n"
        f"{control_csv}"
    )

    print(
        "\nSaved unmatched-control diagnostic:\n"
        f"{unmatched_csv}"
    )


# ============================================================
# 11. MAIN
# ============================================================

def main():

    print("\n" + "=" * 80)
    print(
        "PHARMAAI - BUILD MAPPED "
        "REFERENCE LABELS"
    )
    print("=" * 80)

    print(
        "\nThis script uses the approved "
        "OMOP-to-FAERS outcome mapping."
    )

    print(
        "It does NOT recalculate "
        "ROR, PRR, or IC."
    )

    print(
        "It does NOT treat unmatched "
        "FAERS pairs as negative controls."
    )

    omop = load_omop_reference()

    mapping = load_mapping()

    expanded = expand_reference_controls(
        omop,
        mapping
    )

    summaries = []

    for (
        dataset_name,
        file_path
    ) in SIGNAL_FILES.items():

        signal = load_signal_table(
            dataset_name,
            file_path
        )

        (
            matched,
            matched_controls,
            unmatched_controls,
            summary,
        ) = match_dataset(
            dataset_name,
            signal,
            expanded
        )

        save_dataset_outputs(
            dataset_name,
            matched,
            matched_controls,
            unmatched_controls
        )

        summaries.append(
            summary
        )

        # Free the large signal table before
        # processing the next period.
        del signal

    # --------------------------------------------------------
    # Save combined coverage summary
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        summaries
    )

    summary_file = (
        RESULTS_DIR
        / "mapped_reference_match_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    print("\n" + "=" * 80)
    print("FINAL MATCH SUMMARY")
    print("=" * 80)

    print(
        "\n"
        + summary_df.to_string(
            index=False
        )
    )

    print(
        "\n\nSummary saved:\n"
        f"{summary_file}"
    )

    print("\n" + "=" * 80)
    print("MAPPED REFERENCE LABEL BUILD COMPLETE")
    print("=" * 80)

    print(
        "\nIMPORTANT:"
    )

    print(
        "Before feature engineering, inspect "
        "matched_positive_controls and "
        "matched_negative_controls."
    )

    print(
        "A supervised ML comparison requires "
        "enough usable controls from both "
        "classes."
    )


# ============================================================
# 12. RUN
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\nMapped reference-label build "
            "interrupted."
        )

        sys.exit(1)

    except Exception as error:

        print(
            "\nMapped reference-label build "
            "failed:"
        )

        print(error)

        raise
