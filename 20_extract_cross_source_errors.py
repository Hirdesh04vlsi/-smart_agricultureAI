from pathlib import Path
import pandas as pd
import shutil

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"D:\AI_Agriculture_Dataset\RAW_DATASETS\BOTTLE_GUARD_MASTER")

RESULT_DIR = BASE_DIR / "CROSS_SOURCE_RESULTS" / "HOLDOUT_BG01_Mendeley"
PREDICTIONS_FILE = RESULT_DIR / "test_predictions.csv"

OUTPUT_DIR = RESULT_DIR / "ERROR_ANALYSIS"

# Main confusion pairs we want to inspect
CONFUSION_PAIRS = [
    ("Anthracnose", "Alternaria_Leaf_Blight"),
    ("Anthracnose", "Mosaic_Virus"),
    ("Anthracnose", "Early_Alternaria_Leaf_Blight"),
]

# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("=" * 70)
print("CROSS-SOURCE ERROR EXTRACTION")
print("=" * 70)

df = pd.read_csv(PREDICTIONS_FILE)

print(f"Total test images: {len(df)}")

# Only incorrect predictions
errors = df[df["correct"] == False].copy()

print(f"Total incorrect predictions: {len(errors)}")

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# EXTRACT SELECTED CONFUSION PAIRS
# ============================================================

summary = []

for true_class, predicted_class in CONFUSION_PAIRS:

    pair_name = f"{true_class}_AS_{predicted_class}"

    # Make filesystem-safe folder name
    pair_dir = OUTPUT_DIR / pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)

    pair_df = errors[
        (errors["standardized_class"] == true_class) &
        (errors["predicted_class"] == predicted_class)
    ].copy()

    print()
    print("-" * 70)
    print(f"TRUE      : {true_class}")
    print(f"PREDICTED : {predicted_class}")
    print(f"COUNT     : {len(pair_df)}")

    copied = 0
    failed = 0

    for _, row in pair_df.iterrows():

        source_path = Path(row["master_path"])

        if not source_path.exists():
            print(f"WARNING: Image not found: {source_path}")
            failed += 1
            continue

        # Add confidence to filename
        confidence = float(row["confidence"])

        new_name = (
            f"{row['image_id']}"
            f"_conf_{confidence:.3f}"
            f"_{source_path.name}"
        )

        destination = pair_dir / new_name

        shutil.copy2(source_path, destination)

        copied += 1

    summary.append({
        "true_class": true_class,
        "predicted_class": predicted_class,
        "error_count": len(pair_df),
        "copied": copied,
        "failed": failed,
    })

    print(f"Copied    : {copied}")
    print(f"Failed    : {failed}")

# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(summary)

summary_file = OUTPUT_DIR / "selected_confusion_pairs.csv"
summary_df.to_csv(summary_file, index=False)

print()
print("=" * 70)
print("ERROR EXTRACTION COMPLETED")
print("=" * 70)

print(f"Output directory:")
print(OUTPUT_DIR)

print()
print("Created folders:")

for item in OUTPUT_DIR.iterdir():
    if item.is_dir():
        print(f"  {item.name}")

print()
print(f"Summary saved:")
print(summary_file)