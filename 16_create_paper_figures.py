import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = r"D:\AI_Agriculture_Dataset\RAW_DATASETS\BOTTLE_GUARD_MASTER"

OUTPUT_DIR = os.path.join(BASE_DIR, "PAPER_FIGURES")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# TRAINING HISTORY
# ============================================================

history = pd.DataFrame({
    "Epoch": list(range(1, 16)),

    "Train_Loss": [
        1.1443, 0.5455, 0.3926, 0.3207, 0.2743,
        0.2445, 0.1992, 0.1811, 0.1653, 0.1535,
        0.1297, 0.1226, 0.1196, 0.0946, 0.0892
    ],

    "Train_Accuracy": [
        59.52, 78.13, 83.93, 87.01, 89.03,
        90.17, 91.76, 92.51, 93.47, 93.53,
        94.52, 94.84, 94.63, 95.88, 95.95
    ],

    "Val_Loss": [
        0.6746, 0.4993, 0.4391, 0.3474, 0.3070,
        0.2992, 0.2827, 0.2600, 0.3248, 0.2367,
        0.2527, 0.2536, 0.2604, 0.2442, 0.2475
    ],

    "Val_Accuracy": [
        77.58, 82.94, 85.48, 87.85, 90.06,
        90.72, 91.50, 92.66, 90.72, 92.77,
        92.27, 92.77, 92.55, 92.82, 92.93
    ]
})

history.to_csv(
    os.path.join(OUTPUT_DIR, "training_history.csv"),
    index=False
)

# ============================================================
# FIGURE 1 — ACCURACY
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history["Epoch"],
    history["Train_Accuracy"],
    marker="o",
    linewidth=2,
    label="Training Accuracy"
)

plt.plot(
    history["Epoch"],
    history["Val_Accuracy"],
    marker="o",
    linewidth=2,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "Fig1_training_validation_accuracy.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# FIGURE 2 — LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history["Epoch"],
    history["Train_Loss"],
    marker="o",
    linewidth=2,
    label="Training Loss"
)

plt.plot(
    history["Epoch"],
    history["Val_Loss"],
    marker="o",
    linewidth=2,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "Fig2_training_validation_loss.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# FIGURE 3 — CONFUSION MATRIX
# ============================================================

cm_file = os.path.join(
    BASE_DIR,
    "TEST_EVALUATION",
    "confusion_matrix.csv"
)

cm = pd.read_csv(
    cm_file,
    index_col=0
)

plt.figure(figsize=(11, 9))

plt.imshow(cm.values)

plt.colorbar()

plt.xticks(
    range(len(cm.columns)),
    cm.columns,
    rotation=90
)

plt.yticks(
    range(len(cm.index)),
    cm.index
)

plt.xlabel("Predicted Class")
plt.ylabel("True Class")

plt.title(
    "Confusion Matrix - Bottle Gourd Disease Classification"
)

for i in range(len(cm.index)):
    for j in range(len(cm.columns)):

        plt.text(
            j,
            i,
            str(cm.iloc[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "Fig3_confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# FIGURE 4 — PER CLASS METRICS
# ============================================================

report_file = os.path.join(
    BASE_DIR,
    "TEST_EVALUATION",
    "classification_report.csv"
)

report = pd.read_csv(
    report_file,
    index_col=0
)

classes = [
    c for c in report.index
    if c not in ["accuracy", "macro avg", "weighted avg"]
]

metrics = report.loc[
    classes,
    ["precision", "recall", "f1-score"]
]

x = np.arange(len(classes))
width = 0.25

plt.figure(figsize=(13, 6))

plt.bar(
    x - width,
    metrics["precision"] * 100,
    width,
    label="Precision"
)

plt.bar(
    x,
    metrics["recall"] * 100,
    width,
    label="Recall"
)

plt.bar(
    x + width,
    metrics["f1-score"] * 100,
    width,
    label="F1-score"
)

plt.xticks(
    x,
    classes,
    rotation=75,
    ha="right"
)

plt.ylabel("Score (%)")
plt.title("Per-Class Classification Performance")
plt.legend()
plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "Fig4_per_class_metrics.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# COMPLETE
# ============================================================

print("=" * 70)
print("PAPER FIGURES CREATED")
print("=" * 70)

print(f"Output folder:")
print(OUTPUT_DIR)

for file in sorted(os.listdir(OUTPUT_DIR)):
    print(file)

print("=" * 70)