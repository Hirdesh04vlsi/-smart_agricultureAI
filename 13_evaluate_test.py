import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\AI_Agriculture_Dataset\RAW_DATASETS\BOTTLE_GUARD_MASTER"

TEST_DIR = os.path.join(BASE_DIR, "RESIZED_DATASET", "test")
MODEL_PATH = os.path.join(
    BASE_DIR, "TRAINING_OUTPUT", "best_efficientnet_b0.pth"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "TEST_EVALUATION")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("BOTTLE GOURD DISEASE CLASSIFIER - TEST EVALUATION")
print("=" * 70)
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# TRANSFORM
# ============================================================

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# DATASET
# ============================================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

class_names = test_dataset.classes
num_classes = len(class_names)

print(f"Test images: {len(test_dataset)}")
print(f"Number of classes: {num_classes}")
print("\nClasses:")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")

# ============================================================
# MODEL
# ============================================================

model = models.efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

# Handle either direct state_dict or checkpoint dictionary
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("\nModel loaded successfully.")

# ============================================================
# INFERENCE
# ============================================================

all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

# Convert to numpy
y_true = np.array(all_labels)
y_pred = np.array(all_predictions)
y_prob = np.array(all_probabilities)

# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = accuracy_score(y_true, y_pred)

precision_macro, recall_macro, f1_macro, _ = \
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

precision_weighted, recall_weighted, f1_weighted, _ = \
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

print("\n" + "=" * 70)
print("OVERALL TEST RESULTS")
print("=" * 70)

print(f"Test Accuracy       : {accuracy * 100:.2f}%")
print(f"Macro Precision     : {precision_macro * 100:.2f}%")
print(f"Macro Recall        : {recall_macro * 100:.2f}%")
print(f"Macro F1-score      : {f1_macro * 100:.2f}%")
print(f"Weighted Precision   : {precision_weighted * 100:.2f}%")
print(f"Weighted Recall      : {recall_weighted * 100:.2f}%")
print(f"Weighted F1-score    : {f1_weighted * 100:.2f}%")

# ============================================================
# PER-CLASS REPORT
# ============================================================

report_dict = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report_dict).transpose()

print("\n" + "=" * 70)
print("PER-CLASS RESULTS")
print("=" * 70)

print(
    report_df[
        ["precision", "recall", "f1-score", "support"]
    ].to_string()
)

report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.csv"
)

report_df.to_csv(report_path)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.csv"
)

cm_df.to_csv(cm_path)

# ============================================================
# CONFUSION MATRIX FIGURE
# ============================================================

plt.figure(
    figsize=(12, 10)
)

plt.imshow(cm)

plt.title(
    "Bottle Gourd Disease Classification - Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    np.arange(num_classes),
    class_names,
    rotation=90
)

plt.yticks(
    np.arange(num_classes),
    class_names
)

plt.xlabel("Predicted Class")
plt.ylabel("True Class")

# Write numbers into cells
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

cm_png = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_png,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# CORRECT / INCORRECT
# ============================================================

correct = int(np.sum(y_true == y_pred))
incorrect = int(np.sum(y_true != y_pred))

print("\n" + "=" * 70)
print("PREDICTION COUNTS")
print("=" * 70)

print(f"Correct predictions   : {correct}")
print(f"Incorrect predictions : {incorrect}")
print(f"Total test images     : {len(y_true)}")

# ============================================================
# SAVE SUMMARY JSON
# ============================================================

summary = {
    "model": "EfficientNet-B0",
    "test_images": int(len(y_true)),
    "num_classes": int(num_classes),

    "accuracy": float(accuracy),

    "macro_precision": float(precision_macro),
    "macro_recall": float(recall_macro),
    "macro_f1": float(f1_macro),

    "weighted_precision": float(precision_weighted),
    "weighted_recall": float(recall_weighted),
    "weighted_f1": float(f1_weighted),

    "correct_predictions": correct,
    "incorrect_predictions": incorrect,

    "classes": class_names
}

summary_path = os.path.join(
    OUTPUT_DIR,
    "test_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )

# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_rows = []

for i in range(len(y_true)):

    confidence = float(
        np.max(y_prob[i])
    )

    prediction_rows.append({
        "index": i,
        "true_class": class_names[y_true[i]],
        "predicted_class": class_names[y_pred[i]],
        "confidence": confidence,
        "correct": bool(y_true[i] == y_pred[i])
    })

predictions_df = pd.DataFrame(
    prediction_rows
)

predictions_path = os.path.join(
    OUTPUT_DIR,
    "test_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(report_path)
print(cm_path)
print(cm_png)
print(summary_path)
print(predictions_path)

print("\nTEST EVALUATION COMPLETE.")
print("=" * 70)