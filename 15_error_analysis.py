import os
import pandas as pd
import numpy as np

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\AI_Agriculture_Dataset\RAW_DATASETS\BOTTLE_GUARD_MASTER"

EVAL_DIR = os.path.join(
    BASE_DIR,
    "TEST_EVALUATION"
)

PREDICTIONS_FILE = os.path.join(
    EVAL_DIR,
    "test_predictions.csv"
)

OUTPUT_DIR = os.path.join(
    EVAL_DIR,
    "ERROR_ANALYSIS"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD PREDICTIONS
# ============================================================

df = pd.read_csv(PREDICTIONS_FILE)

print("=" * 75)
print("BOTTLE GOURD MODEL - ERROR ANALYSIS")
print("=" * 75)

print(f"Total test images: {len(df)}")

# ============================================================
# WRONG PREDICTIONS
# ============================================================

wrong = df[df["correct"] == False].copy()

print(f"Incorrect predictions: {len(wrong)}")
print(
    f"Overall error rate: "
    f"{len(wrong) / len(df) * 100:.2f}%"
)

# ============================================================
# 1. ERRORS BY TRUE CLASS
# ============================================================

true_counts = (
    df.groupby("true_class")
    .size()
    .reset_index(name="total_images")
)

wrong_counts = (
    wrong.groupby("true_class")
    .size()
    .reset_index(name="incorrect")
)

error_by_class = true_counts.merge(
    wrong_counts,
    on="true_class",
    how="left"
)

error_by_class["incorrect"] = (
    error_by_class["incorrect"]
    .fillna(0)
    .astype(int)
)

error_by_class["error_rate_percent"] = (
    error_by_class["incorrect"]
    / error_by_class["total_images"]
    * 100
)

error_by_class = error_by_class.sort_values(
    "error_rate_percent",
    ascending=False
)

print("\n" + "=" * 75)
print("ERROR RATE BY TRUE CLASS")
print("=" * 75)

print(
    error_by_class.to_string(index=False)
)

error_by_class.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "error_by_class.csv"
    ),
    index=False
)

# ============================================================
# 2. TOP CONFUSION PAIRS
# ============================================================

confusion_pairs = (
    wrong
    .groupby(
        ["true_class", "predicted_class"]
    )
    .size()
    .reset_index(name="errors")
)

confusion_pairs = confusion_pairs.sort_values(
    "errors",
    ascending=False
)

print("\n" + "=" * 75)
print("TOP CONFUSION PAIRS")
print("=" * 75)

print(
    confusion_pairs.head(20).to_string(
        index=False
    )
)

confusion_pairs.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "top_confusion_pairs.csv"
    ),
    index=False
)

# ============================================================
# 3. WRONG PREDICTIONS BY CONFIDENCE
# ============================================================

wrong["confidence_percent"] = (
    wrong["confidence"] * 100
)

print("\n" + "=" * 75)
print("WRONG PREDICTIONS - CONFIDENCE")
print("=" * 75)

print(
    wrong[
        [
            "true_class",
            "predicted_class",
            "confidence_percent"
        ]
    ]
    .sort_values(
        "confidence_percent",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)

# ============================================================
# 4. HIGH-CONFIDENCE WRONG PREDICTIONS
# ============================================================

high_conf_wrong = wrong[
    wrong["confidence"] >= 0.80
].copy()

print("\n" + "=" * 75)
print("HIGH-CONFIDENCE WRONG PREDICTIONS")
print("=" * 75)

print(
    f"Wrong predictions with >=80% confidence: "
    f"{len(high_conf_wrong)}"
)

if len(high_conf_wrong) > 0:

    print(
        high_conf_wrong[
            [
                "true_class",
                "predicted_class",
                "confidence"
            ]
        ]
        .sort_values(
            "confidence",
            ascending=False
        )
        .head(30)
        .to_string(index=False)
    )

high_conf_wrong.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "high_confidence_wrong_predictions.csv"
    ),
    index=False
)

# ============================================================
# 5. LOW-CONFIDENCE WRONG PREDICTIONS
# ============================================================

low_conf_wrong = wrong[
    wrong["confidence"] < 0.50
].copy()

print("\n" + "=" * 75)
print("LOW-CONFIDENCE WRONG PREDICTIONS")
print("=" * 75)

print(
    f"Wrong predictions with <50% confidence: "
    f"{len(low_conf_wrong)}"
)

low_conf_wrong.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "low_confidence_wrong_predictions.csv"
    ),
    index=False
)

# ============================================================
# 6. CONFUSION MATRIX STYLE TABLE
# ============================================================

confusion_matrix = pd.crosstab(
    df["true_class"],
    df["predicted_class"]
)

confusion_matrix.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "error_analysis_confusion_matrix.csv"
    )
)

# ============================================================
# 7. SUMMARY
# ============================================================

summary = {
    "total_test_images": len(df),
    "incorrect_predictions": len(wrong),
    "error_rate_percent": len(wrong) / len(df) * 100,
    "high_confidence_wrong_ge_80_percent": len(
        high_conf_wrong
    ),
    "low_confidence_wrong_lt_50_percent": len(
        low_conf_wrong
    )
}

summary_df = pd.DataFrame(
    [summary]
)

summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "error_analysis_summary.csv"
    ),
    index=False
)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("FILES SAVED")
print("=" * 75)

print(
    f"Folder:\n{OUTPUT_DIR}"
)

print("\nError analysis complete.")
print("=" * 75)