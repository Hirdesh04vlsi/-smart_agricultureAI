import os
import pandas as pd
from pathlib import Path

# ============================================================
# STEP 18
# CROSS-SOURCE / LEAVE-ONE-DATASET-OUT EXPERIMENT
#
# Uses:
#   master_dataset_metadata.csv
#
# Important:
#   - dataset     = source dataset name
#   - master_path = deduplicated usable image
#
# No images are copied.
# Only CSV manifests are created.
# ============================================================


# ------------------------------------------------------------
# 1. BASE DIRECTORIES
# ------------------------------------------------------------

BASE_DIR = Path(
    r"D:\AI_Agriculture_Dataset\RAW_DATASETS\BOTTLE_GUARD_MASTER"
)

METADATA_FILE = BASE_DIR / "master_dataset_metadata.csv"

OUTPUT_DIR = BASE_DIR / "CROSS_SOURCE_DATASET"


# ------------------------------------------------------------
# 2. CHECK INPUT
# ------------------------------------------------------------

if not METADATA_FILE.exists():
    raise FileNotFoundError(
        f"\nERROR: Metadata file not found:\n{METADATA_FILE}"
    )


if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"""
ERROR: CROSS_SOURCE_DATASET already exists.

Delete it manually first:

Remove-Item -Recurse -Force ".\\CROSS_SOURCE_DATASET"

Then run this script again.
"""
    )


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 3. LOAD MASTER METADATA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 18: CROSS-SOURCE DATASET PREPARATION")
print("=" * 70)

print("\nLoading master metadata...")

df = pd.read_csv(METADATA_FILE)

print(f"Total metadata records: {len(df):,}")


# ------------------------------------------------------------
# 4. CHECK REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "image_id",
    "dataset",
    "original_class",
    "standardized_class",
    "filename",
    "original_filename",
    "source_path",
    "master_path",
    "sha256",
    "augmented",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "\nERROR: Missing required columns:\n"
        + "\n".join(missing_columns)
    )


print("\nRequired metadata columns found.")


# ------------------------------------------------------------
# 5. SHOW SOURCE DATASETS
# ------------------------------------------------------------

sources = sorted(df["dataset"].dropna().unique())

print("\nSource datasets found:")

for source in sources:
    count = len(df[df["dataset"] == source])
    print(f"  {source:<25} {count:>6,} images")

print(f"\nTotal source datasets: {len(sources)}")


# ------------------------------------------------------------
# 6. CHECK MASTER IMAGE PATHS
# ------------------------------------------------------------

print("\nChecking master image paths...")

missing_paths = 0

for path in df["master_path"]:
    path = Path(str(path))

    if not path.is_absolute():
        path = BASE_DIR / path

    if not path.exists():
        missing_paths += 1

if missing_paths > 0:
    raise FileNotFoundError(
        f"\nERROR: {missing_paths} master image paths do not exist."
    )

print("All master image paths are valid.")


# ------------------------------------------------------------
# 7. CLASS DISTRIBUTION BY SOURCE
# ------------------------------------------------------------

print("\nCreating source/class distribution...")

source_class = (
    df.groupby(
        ["dataset", "standardized_class"]
    )
    .size()
    .reset_index(name="image_count")
)

source_class.to_csv(
    OUTPUT_DIR / "source_class_distribution.csv",
    index=False
)


# ------------------------------------------------------------
# 8. PIVOT TABLE
# ------------------------------------------------------------

pivot = (
    source_class
    .pivot(
        index="standardized_class",
        columns="dataset",
        values="image_count"
    )
    .fillna(0)
    .astype(int)
)

pivot.to_csv(
    OUTPUT_DIR / "source_class_distribution_pivot.csv"
)


# ------------------------------------------------------------
# 9. ALL STANDARDIZED CLASSES
# ------------------------------------------------------------

all_classes = sorted(
    df["standardized_class"]
    .dropna()
    .unique()
)

print("\nStandardized classes:")

for cls in all_classes:
    total = len(
        df[df["standardized_class"] == cls]
    )
    print(f"  {cls:<35} {total:>6,}")


# ------------------------------------------------------------
# 10. CREATE LEAVE-ONE-SOURCE-OUT FOLDS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CREATING LEAVE-ONE-SOURCE-OUT FOLDS")
print("=" * 70)


fold_summary = []


for held_out_source in sources:

    print("\n" + "-" * 70)
    print(f"Held-out source: {held_out_source}")
    print("-" * 70)


    # --------------------------------------------------------
    # TEST = HELD-OUT SOURCE
    # TRAIN = ALL OTHER SOURCES
    # --------------------------------------------------------

    test_df = df[
        df["dataset"] == held_out_source
    ].copy()

    train_pool_df = df[
        df["dataset"] != held_out_source
    ].copy()


    # --------------------------------------------------------
    # FIND COMMON CLASSES
    #
    # A class can only be evaluated fairly if:
    #   1. It exists in training data
    #   2. It exists in held-out test source
    # --------------------------------------------------------

    train_classes = set(
        train_pool_df["standardized_class"]
    )

    test_classes = set(
        test_df["standardized_class"]
    )

    common_classes = sorted(
        train_classes.intersection(test_classes)
    )


    print("\nClasses present in held-out source:")

    for cls in sorted(test_classes):
        count = len(
            test_df[
                test_df["standardized_class"] == cls
            ]
        )
        print(f"  {cls:<35} {count:>6,}")


    print("\nCommon train/test classes:")

    for cls in common_classes:
        train_count = len(
            train_pool_df[
                train_pool_df["standardized_class"] == cls
            ]
        )

        test_count = len(
            test_df[
                test_df["standardized_class"] == cls
            ]
        )

        print(
            f"  {cls:<35}"
            f" train={train_count:>6,}"
            f"  test={test_count:>6,}"
        )


    # --------------------------------------------------------
    # KEEP ONLY COMMON CLASSES
    # --------------------------------------------------------

    train_fold = train_pool_df[
        train_pool_df["standardized_class"].isin(
            common_classes
        )
    ].copy()

    test_fold = test_df[
        test_df["standardized_class"].isin(
            common_classes
        )
    ].copy()


    # --------------------------------------------------------
    # CREATE FOLD DIRECTORY
    # --------------------------------------------------------

    safe_name = held_out_source.replace(
        " ", "_"
    )

    fold_dir = (
        OUTPUT_DIR /
        f"HOLDOUT_{safe_name}"
    )

    fold_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # SAVE TRAIN / TEST MANIFESTS
    # --------------------------------------------------------

    train_fold.to_csv(
        fold_dir / "train_pool.csv",
        index=False
    )

    test_fold.to_csv(
        fold_dir / "test.csv",
        index=False
    )


    # --------------------------------------------------------
    # SAVE CLASS DISTRIBUTION
    # --------------------------------------------------------

    train_distribution = (
        train_fold
        .groupby("standardized_class")
        .size()
        .reset_index(
            name="train_count"
        )
    )

    test_distribution = (
        test_fold
        .groupby("standardized_class")
        .size()
        .reset_index(
            name="test_count"
        )
    )


    distribution = pd.merge(
        train_distribution,
        test_distribution,
        on="standardized_class",
        how="outer"
    ).fillna(0)


    distribution[
        ["train_count", "test_count"]
    ] = distribution[
        ["train_count", "test_count"]
    ].astype(int)


    distribution.to_csv(
        fold_dir / "class_distribution.csv",
        index=False
    )


    # --------------------------------------------------------
    # SAVE FOLD SUMMARY
    # --------------------------------------------------------

    summary = {
        "held_out_source": held_out_source,
        "train_images": len(train_fold),
        "test_images": len(test_fold),
        "number_of_common_classes": len(common_classes),
        "common_classes": ", ".join(common_classes),
    }

    fold_summary.append(summary)


    print("\nFold created successfully.")

    print(
        f"  Training images : {len(train_fold):,}"
    )

    print(
        f"  Test images     : {len(test_fold):,}"
    )

    print(
        f"  Common classes  : {len(common_classes)}"
    )

    print(
        f"  Output          : {fold_dir}"
    )


# ------------------------------------------------------------
# 11. SAVE OVERALL FOLD SUMMARY
# ------------------------------------------------------------

fold_summary_df = pd.DataFrame(
    fold_summary
)

fold_summary_df.to_csv(
    OUTPUT_DIR / "cross_source_fold_summary.csv",
    index=False
)


# ------------------------------------------------------------
# 12. SAVE COMPLETE CONFIGURATION
# ------------------------------------------------------------

config = {
    "experiment": "Leave-One-Source-Out Cross-Dataset Generalization",
    "metadata_file": str(METADATA_FILE),
    "total_master_images": len(df),
    "number_of_sources": len(sources),
    "sources": ", ".join(sources),
    "number_of_classes": len(all_classes),
    "classes": ", ".join(all_classes),
    "image_path_column": "master_path",
    "source_column": "dataset",
    "note": (
        "Only classes present in both the training pool "
        "and held-out source are included in each fold."
    ),
}

pd.DataFrame(
    [config]
).to_csv(
    OUTPUT_DIR / "experiment_config.csv",
    index=False
)


# ------------------------------------------------------------
# 13. FINAL OUTPUT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 18 COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nOutput directory:\n{OUTPUT_DIR}"
)

print("\nCreated files:")

print("  source_class_distribution.csv")
print("  source_class_distribution_pivot.csv")
print("  cross_source_fold_summary.csv")
print("  experiment_config.csv")

print("\nCreated folds:")

for source in sources:
    safe_name = source.replace(
        " ", "_"
    )

    print(
        f"  HOLDOUT_{safe_name}"
    )

print("\nEach fold contains:")
print("  train_pool.csv")
print("  test.csv")
print("  class_distribution.csv")

print("\nNo raw dataset files were modified.")
print("No images were copied.")
print("No images were deleted.")

print("\nNext step will be training the cross-source model.")
print("=" * 70)