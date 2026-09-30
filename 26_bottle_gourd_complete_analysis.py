import os
import json
import math
import shutil
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

from sklearn.preprocessing import label_binarize


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(
    r"D:\AI_Agriculture_Dataset\RAW_DATASETS\BOTTLE_GUARD_MASTER"
)

TRAINING_DIR = ROOT / "TRAINING_OUTPUT"

TEST_DIR = ROOT / "TEST_EVALUATION"

ERROR_DIR = TEST_DIR / "ERROR_ANALYSIS"

PAPER_DIR = ROOT / "PAPER_FIGURES"

OUTPUT = ROOT / "BOTTLE_GOURD_COMPLETE_RESULTS"


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

DIRS = [

    OUTPUT,

    OUTPUT / "01_TRAINING",

    OUTPUT / "02_TEST_METRICS",

    OUTPUT / "03_CONFUSION_MATRICES",

    OUTPUT / "04_CLASS_METRICS",

    OUTPUT / "05_CONFIDENCE_ANALYSIS",

    OUTPUT / "06_ERROR_ANALYSIS",

    OUTPUT / "07_ROC_PR_CURVES",

    OUTPUT / "08_CORRECT_IMAGES",

    OUTPUT / "09_WRONG_IMAGES",

    OUTPUT / "10_HIGH_CONFIDENCE_WRONG",

    OUTPUT / "11_LOW_CONFIDENCE_WRONG",

    OUTPUT / "12_CLASS_IMAGE_GALLERIES",

    OUTPUT / "13_CROSS_SOURCE",

    OUTPUT / "14_PUBLICATION_FIGURES",

    OUTPUT / "15_REPORT"

]


for d in DIRS:

    d.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# PLOT CONFIGURATION
# ============================================================

plt.rcParams.update({

    "figure.dpi": 120,

    "savefig.dpi": 300,

    "axes.grid": True,

    "font.size": 10

})


# ============================================================
# HELPERS
# ============================================================

def save_fig(path):

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def safe_name(text):

    return "".join(

        c if c.isalnum()
        else "_"

        for c in str(text)

    )


def get_font(size=16):

    fonts = [

        r"C:\Windows\Fonts\arial.ttf",

        r"C:\Windows\Fonts\segoeui.ttf",

        r"C:\Windows\Fonts\calibri.ttf"

    ]

    for f in fonts:

        if os.path.exists(f):

            try:

                return ImageFont.truetype(
                    f,
                    size
                )

            except:
                pass

    return ImageFont.load_default()


def find_existing(*paths):

    for path in paths:

        if path.exists():

            return path

    return None


# ============================================================
# LOCATE FILES
# ============================================================

print("=" * 80)

print("BOTTLE GOURD COMPLETE MODEL ANALYSIS")

print("=" * 80)

print()


history_file = find_existing(

    TRAINING_DIR /
    "training_history.json",

    TRAINING_DIR /
    "training_history.csv",

    PAPER_DIR /
    "training_history.csv"

)


predictions_file = find_existing(

    TEST_DIR /
    "test_predictions.csv"

)


report_file = find_existing(

    TEST_DIR /
    "classification_report.csv"

)


confusion_file = find_existing(

    TEST_DIR /
    "confusion_matrix.csv"

)


summary_file = find_existing(

    TEST_DIR /
    "test_summary.json"

)


if predictions_file is None:

    raise FileNotFoundError(

        "test_predictions.csv was not found:\n"
        + str(TEST_DIR)

    )


print(
    "Predictions:",
    predictions_file
)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

pred = pd.read_csv(
    predictions_file
)


print(
    "\nPrediction columns:"
)

print(
    list(pred.columns)
)


# ============================================================
# DETECT COLUMN NAMES
# ============================================================

def find_column(
    dataframe,
    possible
):

    for col in possible:

        if col in dataframe.columns:

            return col

    return None


TRUE_COL = find_column(

    pred,

    [
        "true_class",
        "true_label",
        "actual",
        "actual_class",
        "y_true",
        "label"
    ]

)


PRED_COL = find_column(

    pred,

    [
        "predicted_class",
        "predicted_label",
        "prediction",
        "pred_class",
        "y_pred"
    ]

)


CONF_COL = find_column(

    pred,

    [
        "confidence",
        "confidence_score",
        "max_probability",
        "probability",
        "score"
    ]

)


PATH_COL = find_column(

    pred,

    [
        "image_path",
        "path",
        "full_path",
        "source_path",
        "master_path",
        "filename"
    ]

)


if TRUE_COL is None:

    raise ValueError(
        "Could not identify TRUE label column."
    )


if PRED_COL is None:

    raise ValueError(
        "Could not identify PREDICTED label column."
    )


# ============================================================
# NORMALIZE CONFIDENCE
# ============================================================

if CONF_COL is not None:

    confidence = pd.to_numeric(

        pred[CONF_COL],

        errors="coerce"

    )

    # Convert percentage to 0-1
    if confidence.max() > 1:

        confidence = confidence / 100.0

else:

    confidence = pd.Series(
        np.nan,
        index=pred.index
    )


pred["_confidence"] = confidence


# ============================================================
# CORRECT / WRONG
# ============================================================

pred["_correct"] = (

    pred[TRUE_COL].astype(str)
    ==
    pred[PRED_COL].astype(str)

)


# ============================================================
# CLASS LIST
# ============================================================

classes = sorted(

    set(
        pred[TRUE_COL].astype(str)
    )
    |
    set(
        pred[PRED_COL].astype(str)
    )

)


print(
    "\nClasses:"
)

for i, cls in enumerate(classes):

    print(
        f"{i}: {cls}"
    )


# ============================================================
# BASIC METRICS
# ============================================================

y_true = pred[
    TRUE_COL
].astype(str)


y_pred = pred[
    PRED_COL
].astype(str)


accuracy = (
    pred["_correct"].mean()
)


report_dict = classification_report(

    y_true,

    y_pred,

    labels=classes,

    output_dict=True,

    zero_division=0

)


report_df = pd.DataFrame(
    report_dict
).T


report_df.to_csv(

    OUTPUT /
    "02_TEST_METRICS" /
    "classification_report_complete.csv"

)


# ============================================================
# MASTER SUMMARY JSON
# ============================================================

summary = {

    "crop":
        "Bottle Gourd",

    "total_test_images":
        int(len(pred)),

    "correct":
        int(pred["_correct"].sum()),

    "incorrect":
        int((~pred["_correct"]).sum()),

    "accuracy":
        float(accuracy),

    "macro_precision":
        float(
            report_dict[
                "macro avg"
            ]["precision"]
        ),

    "macro_recall":
        float(
            report_dict[
                "macro avg"
            ]["recall"]
        ),

    "macro_f1":
        float(
            report_dict[
                "macro avg"
            ]["f1-score"]
        ),

    "weighted_precision":
        float(
            report_dict[
                "weighted avg"
            ]["precision"]
        ),

    "weighted_recall":
        float(
            report_dict[
                "weighted avg"
            ]["recall"]
        ),

    "weighted_f1":
        float(
            report_dict[
                "weighted avg"
            ]["f1-score"]
        )

}


with open(

    OUTPUT /
    "02_TEST_METRICS" /
    "master_metrics.json",

    "w",

    encoding="utf-8"

) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# 1. TRAINING HISTORY
# ============================================================

print(
    "\nCreating training analysis..."
)


if history_file:

    if history_file.suffix.lower() == ".json":

        with open(
            history_file,
            "r",
            encoding="utf-8"
        ) as f:

            history = json.load(f)


        hist = pd.DataFrame(
            history
        )


    else:

        hist = pd.read_csv(
            history_file
        )


    hist.to_csv(

        OUTPUT /
        "01_TRAINING" /
        "training_history.csv",

        index=False

    )


    # --------------------------------------------------------
    # FIND ACCURACY COLUMNS
    # --------------------------------------------------------

    train_acc_col = find_column(

        hist,

        [
            "train_accuracy",
            "training_accuracy",
            "train_acc"
        ]

    )


    val_acc_col = find_column(

        hist,

        [
            "val_accuracy",
            "validation_accuracy",
            "val_acc"
        ]

    )


    train_loss_col = find_column(

        hist,

        [
            "train_loss",
            "training_loss"
        ]

    )


    val_loss_col = find_column(

        hist,

        [
            "val_loss",
            "validation_loss"
        ]

    )


    epoch_col = find_column(

        hist,

        [
            "epoch",
            "Epoch"
        ]

    )


    if epoch_col is None:

        hist["epoch"] = np.arange(
            1,
            len(hist) + 1
        )

        epoch_col = "epoch"


    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    if (
        train_acc_col
        and
        val_acc_col
    ):

        plt.figure(
            figsize=(9, 6)
        )


        plt.plot(

            hist[epoch_col],

            hist[train_acc_col],

            marker="o",

            label="Training Accuracy"

        )


        plt.plot(

            hist[epoch_col],

            hist[val_acc_col],

            marker="o",

            label="Validation Accuracy"

        )


        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "Accuracy (%)"
        )

        plt.title(
            "Bottle Gourd EfficientNet-B0 Training and Validation Accuracy"
        )

        plt.legend()

        save_fig(

            OUTPUT /
            "01_TRAINING" /
            "training_validation_accuracy.png"

        )


    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    if (
        train_loss_col
        and
        val_loss_col
    ):

        plt.figure(
            figsize=(9, 6)
        )


        plt.plot(

            hist[epoch_col],

            hist[train_loss_col],

            marker="o",

            label="Training Loss"

        )


        plt.plot(

            hist[epoch_col],

            hist[val_loss_col],

            marker="o",

            label="Validation Loss"

        )


        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "Loss"
        )

        plt.title(
            "Bottle Gourd EfficientNet-B0 Training and Validation Loss"
        )

        plt.legend()

        save_fig(

            OUTPUT /
            "01_TRAINING" /
            "training_validation_loss.png"

        )


# ============================================================
# 2. CONFUSION MATRIX
# ============================================================

print(
    "Creating confusion matrices..."
)


cm = confusion_matrix(

    y_true,

    y_pred,

    labels=classes

)


cm_df = pd.DataFrame(

    cm,

    index=classes,

    columns=classes

)


cm_df.to_csv(

    OUTPUT /
    "03_CONFUSION_MATRICES" /
    "confusion_matrix.csv"

)


# ------------------------------------------------------------
# RAW CONFUSION MATRIX
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 10)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Bottle Gourd Disease Classification - Confusion Matrix"
)

plt.colorbar()

ticks = np.arange(
    len(classes)
)

plt.xticks(
    ticks,
    classes,
    rotation=45,
    ha="right"
)

plt.yticks(
    ticks,
    classes
)


threshold = (
    cm.max() / 2
)


for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        plt.text(

            j,
            i,

            str(cm[i, j]),

            ha="center",

            va="center"

        )


plt.ylabel(
    "True Class"
)

plt.xlabel(
    "Predicted Class"
)

save_fig(

    OUTPUT /
    "03_CONFUSION_MATRICES" /
    "confusion_matrix.png"

)


# ============================================================
# NORMALIZED CONFUSION MATRIX
# ============================================================

cm_normalized = (

    cm.astype(float)
    /
    cm.sum(
        axis=1,
        keepdims=True
    )

)


cm_normalized = np.nan_to_num(
    cm_normalized
)


plt.figure(
    figsize=(12, 10)
)

plt.imshow(
    cm_normalized,
    interpolation="nearest"
)

plt.title(
    "Bottle Gourd - Normalized Confusion Matrix"
)

plt.colorbar(
    label="Proportion"
)


plt.xticks(
    ticks,
    classes,
    rotation=45,
    ha="right"
)

plt.yticks(
    ticks,
    classes
)


for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        plt.text(

            j,
            i,

            f"{cm_normalized[i,j]:.2f}",

            ha="center",

            va="center"

        )


plt.ylabel(
    "True Class"
)

plt.xlabel(
    "Predicted Class"
)

save_fig(

    OUTPUT /
    "03_CONFUSION_MATRICES" /
    "normalized_confusion_matrix.png"

)


# ============================================================
# 3. CLASS METRICS
# ============================================================

class_report = report_df.loc[
    classes
].copy()


class_report[
    [
        "precision",
        "recall",
        "f1-score"
    ]
].to_csv(

    OUTPUT /
    "04_CLASS_METRICS" /
    "per_class_metrics.csv"

)


# ------------------------------------------------------------
# PRECISION RECALL F1
# ------------------------------------------------------------

metrics = [

    "precision",
    "recall",
    "f1-score"

]


x = np.arange(
    len(classes)
)

width = 0.25


plt.figure(
    figsize=(14, 7)
)


for i, metric in enumerate(
    metrics
):

    plt.bar(

        x + (
            i - 1
        ) * width,

        class_report[
            metric
        ],

        width,

        label=metric

    )


plt.xticks(
    x,
    classes,
    rotation=45,
    ha="right"
)

plt.ylim(
    0,
    1.05
)

plt.ylabel(
    "Score"
)

plt.xlabel(
    "Disease Class"
)

plt.title(
    "Per-Class Precision, Recall and F1-Score"
)

plt.legend()

save_fig(

    OUTPUT /
    "04_CLASS_METRICS" /
    "per_class_precision_recall_f1.png"

)


# ============================================================
# SUPPORT
# ============================================================

support = (

    class_report[
        "support"
    ]
    .astype(int)

)


plt.figure(
    figsize=(13, 7)
)

plt.bar(
    classes,
    support
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.xlabel(
    "Class"
)

plt.ylabel(
    "Test Images"
)

plt.title(
    "Test Set Class Support"
)

save_fig(

    OUTPUT /
    "04_CLASS_METRICS" /
    "test_class_support.png"

)


# ============================================================
# 4. CONFIDENCE ANALYSIS
# ============================================================

confidence_valid = pred[
    pred["_confidence"].notna()
].copy()


if len(confidence_valid) > 0:


    # --------------------------------------------------------
    # OVERALL
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    plt.hist(

        confidence_valid[
            "_confidence"
        ],

        bins=20

    )

    plt.xlabel(
        "Model Score"
    )

    plt.ylabel(
        "Number of Images"
    )

    plt.title(
        "Test Prediction Score Distribution"
    )

    save_fig(

        OUTPUT /
        "05_CONFIDENCE_ANALYSIS" /
        "overall_confidence_distribution.png"

    )


    # --------------------------------------------------------
    # CORRECT VS WRONG
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )


    plt.hist(

        confidence_valid.loc[
            confidence_valid["_correct"],
            "_confidence"
        ],

        bins=20,

        alpha=0.7,

        label="Correct"

    )


    plt.hist(

        confidence_valid.loc[
            ~confidence_valid["_correct"],
            "_confidence"
        ],

        bins=20,

        alpha=0.7,

        label="Incorrect"

    )


    plt.xlabel(
        "Model Score"
    )

    plt.ylabel(
        "Number of Images"
    )

    plt.title(
        "Confidence Distribution: Correct vs Incorrect"
    )

    plt.legend()

    save_fig(

        OUTPUT /
        "05_CONFIDENCE_ANALYSIS" /
        "correct_vs_wrong_confidence.png"

    )


    # --------------------------------------------------------
    # WRONG CONFIDENCE
    # --------------------------------------------------------

    wrong = confidence_valid[
        ~confidence_valid["_correct"]
    ]


    wrong.to_csv(

        OUTPUT /
        "05_CONFIDENCE_ANALYSIS" /
        "wrong_predictions_with_confidence.csv",

        index=False

    )


    high_wrong = wrong[
        wrong["_confidence"] >= 0.80
    ]


    low_wrong = wrong[
        wrong["_confidence"] < 0.50
    ]


    high_wrong.to_csv(

        OUTPUT /
        "10_HIGH_CONFIDENCE_WRONG" /
        "high_confidence_wrong.csv",

        index=False

    )


    low_wrong.to_csv(

        OUTPUT /
        "11_LOW_CONFIDENCE_WRONG" /
        "low_confidence_wrong.csv",

        index=False

    )


# ============================================================
# 5. ERROR BY CLASS
# ============================================================

error_rows = []


for cls in classes:

    subset = pred[
        pred[TRUE_COL].astype(str)
        == cls
    ]


    errors = (
        ~subset["_correct"]
    ).sum()


    total = len(
        subset
    )


    rate = (

        errors / total

        if total > 0

        else 0

    )


    error_rows.append({

        "class":
            cls,

        "total":
            total,

        "errors":
            int(errors),

        "correct":
            int(total - errors),

        "error_rate":
            rate

    })


error_df = pd.DataFrame(
    error_rows
)


error_df.to_csv(

    OUTPUT /
    "06_ERROR_ANALYSIS" /
    "error_by_class.csv",

    index=False

)


plt.figure(
    figsize=(13, 7)
)

plt.bar(

    error_df["class"],

    error_df["error_rate"] * 100

)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.xlabel(
    "True Class"
)

plt.ylabel(
    "Error Rate (%)"
)

plt.title(
    "Bottle Gourd Test Error Rate by Class"
)

save_fig(

    OUTPUT /
    "06_ERROR_ANALYSIS" /
    "error_rate_by_class.png"

)


# ============================================================
# 6. TOP CONFUSION PAIRS
# ============================================================

pair_counter = Counter()


for _, row in pred.iterrows():

    true_cls = str(
        row[TRUE_COL]
    )

    pred_cls = str(
        row[PRED_COL]
    )


    if true_cls != pred_cls:

        pair_counter[
            (
                true_cls,
                pred_cls
            )
        ] += 1


pairs = pair_counter.most_common()


pair_df = pd.DataFrame(

    [

        {

            "true_class":
                p[0][0],

            "predicted_class":
                p[0][1],

            "count":
                p[1]

        }

        for p in pairs

    ]

)


pair_df.to_csv(

    OUTPUT /
    "06_ERROR_ANALYSIS" /
    "top_confusion_pairs.csv",

    index=False

)


if len(pair_df) > 0:

    top_pairs = pair_df.head(
        15
    )


    labels = [

        f"{r.true_class}\n→\n{r.predicted_class}"

        for r in top_pairs.itertuples()

    ]


    plt.figure(
        figsize=(14, 8)
    )


    plt.bar(

        labels,

        top_pairs["count"]

    )


    plt.ylabel(
        "Number of Errors"
    )

    plt.xlabel(
        "Confusion Pair"
    )

    plt.title(
        "Top Confusion Pairs"
    )


    save_fig(

        OUTPUT /
        "06_ERROR_ANALYSIS" /
        "top_confusion_pairs.png"

    )


# ============================================================
# 7. ROC CURVES
# ============================================================

print(
    "Creating ROC/PR analysis..."
)


# If per-class probabilities are present
probability_columns = [

    c

    for c in pred.columns

    if c.startswith(
        "prob_"
    )

]


if len(probability_columns) == len(classes):


    probability_columns = sorted(
        probability_columns
    )


    probabilities = pred[
        probability_columns
    ].values


    y_bin = label_binarize(

        y_true,

        classes=classes

    )


    plt.figure(
        figsize=(10, 8)
    )


    for i, cls in enumerate(
        classes
    ):


        fpr, tpr, _ = roc_curve(

            y_bin[:, i],

            probabilities[:, i]

        )


        roc_auc = auc(
            fpr,
            tpr
        )


        plt.plot(

            fpr,

            tpr,

            label=f"{cls} (AUC={roc_auc:.3f})"

        )


    plt.plot(

        [0, 1],

        [0, 1],

        linestyle="--"

    )


    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "Bottle Gourd One-vs-Rest ROC Curves"
    )

    plt.legend(
        fontsize=8
    )


    save_fig(

        OUTPUT /
        "07_ROC_PR_CURVES" /
        "roc_curves.png"

    )


    # --------------------------------------------------------
    # PR CURVES
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 8)
    )


    for i, cls in enumerate(
        classes
    ):


        precision, recall, _ = (
            precision_recall_curve(

                y_bin[:, i],

                probabilities[:, i]

            )
        )


        ap = average_precision_score(

            y_bin[:, i],

            probabilities[:, i]

        )


        plt.plot(

            recall,

            precision,

            label=f"{cls} (AP={ap:.3f})"

        )


    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Bottle Gourd Precision-Recall Curves"
    )

    plt.legend(
        fontsize=8
    )


    save_fig(

        OUTPUT /
        "07_ROC_PR_CURVES" /
        "precision_recall_curves.png"

    )


else:

    print(

        "\nNOTE:"
        "\nPer-class probability columns were not found."
        "\nROC/PR curves cannot be reconstructed from top-1"
        "\npredictions alone."

    )


# ============================================================
# 8. COPY / GALLERY IMAGE FUNCTION
# ============================================================

def make_gallery(

    dataframe,

    output_file,

    title,

    max_images=25

):

    if PATH_COL is None:

        return


    valid = dataframe[
        dataframe[PATH_COL]
        .notna()
    ].copy()


    if len(valid) == 0:

        return


    if len(valid) > max_images:

        valid = valid.head(
            max_images
        )


    thumb_w = 220

    thumb_h = 220

    label_h = 65

    cols = 4

    rows = math.ceil(
        len(valid) / cols
    )


    sheet = Image.new(

        "RGB",

        (
            cols * thumb_w,

            rows *
            (
                thumb_h +
                label_h
            )
            + 50
        ),

        "white"

    )


    draw = ImageDraw.Draw(
        sheet
    )


    title_font = get_font(
        20
    )

    small_font = get_font(
        11
    )


    draw.text(

        (10, 10),

        title,

        fill="black",

        font=title_font

    )


    for i, (_, row) in enumerate(
        valid.iterrows()
    ):


        path = Path(
            str(
                row[PATH_COL]
            )
        )


        if not path.exists():

            continue


        try:

            img = Image.open(
                path
            ).convert(
                "RGB"
            )


            img.thumbnail(

                (
                    thumb_w - 10,

                    thumb_h - 10

                )

            )


            x = (
                i % cols
            ) * thumb_w


            y = (
                i // cols
            ) * (
                thumb_h +
                label_h
            ) + 45


            tile = Image.new(

                "RGB",

                (
                    thumb_w,
                    thumb_h
                ),

                "#eeeeee"

            )


            px = (
                thumb_w -
                img.width
            ) // 2


            py = (
                thumb_h -
                img.height
            ) // 2


            tile.paste(

                img,

                (
                    px,
                    py
                )

            )


            sheet.paste(

                tile,

                (
                    x,
                    y
                )

            )


            true_label = str(
                row[TRUE_COL]
            )

            pred_label = str(
                row[PRED_COL]
            )

            score = row[
                "_confidence"
            ]


            text = (

                f"True: {true_label}\n"
                f"Pred: {pred_label}\n"
                f"Score: {score:.3f}"

                if not pd.isna(score)

                else

                f"True: {true_label}\n"
                f"Pred: {pred_label}"

            )


            draw.multiline_text(

                (
                    x + 5,

                    y + thumb_h + 4

                ),

                text,

                fill="black",

                font=small_font,

                spacing=2

            )


        except Exception:

            continue


    sheet.save(

        output_file,

        quality=95

    )


# ============================================================
# 9. CORRECT IMAGE GALLERY
# ============================================================

correct = pred[
    pred["_correct"]
].copy()


correct = correct.sort_values(

    "_confidence",

    ascending=False

)


make_gallery(

    correct,

    OUTPUT /
    "08_CORRECT_IMAGES" /
    "correct_high_confidence_gallery.jpg",

    "Correctly Classified Bottle Gourd Images",

    25

)


# ============================================================
# 10. WRONG IMAGE GALLERY
# ============================================================

wrong = pred[
    ~pred["_correct"]
].copy()


wrong = wrong.sort_values(

    "_confidence",

    ascending=False

)


make_gallery(

    wrong,

    OUTPUT /
    "09_WRONG_IMAGES" /
    "all_wrong_predictions.jpg",

    "Misclassified Bottle Gourd Images",

    30

)


# ============================================================
# 11. HIGH CONFIDENCE WRONG
# ============================================================

high_wrong = wrong[
    wrong["_confidence"] >= 0.80
]


make_gallery(

    high_wrong,

    OUTPUT /
    "10_HIGH_CONFIDENCE_WRONG" /
    "high_confidence_wrong_gallery.jpg",

    "High-Confidence Incorrect Predictions",

    30

)


# ============================================================
# 12. LOW CONFIDENCE WRONG
# ============================================================

low_wrong = wrong[
    wrong["_confidence"] < 0.50
]


make_gallery(

    low_wrong,

    OUTPUT /
    "11_LOW_CONFIDENCE_WRONG" /
    "low_confidence_wrong_gallery.jpg",

    "Low-Confidence Incorrect Predictions",

    30

)


# ============================================================
# 13. PER-CLASS IMAGE GALLERIES
# ============================================================

for cls in classes:


    cls_data = pred[
        pred[TRUE_COL].astype(str)
        == cls
    ]


    make_gallery(

        cls_data,

        OUTPUT /
        "12_CLASS_IMAGE_GALLERIES" /
        f"{safe_name(cls)}.jpg",

        f"Test Images - {cls}",

        20

    )


# ============================================================
# 14. ERROR CSV
# ============================================================

wrong.to_csv(

    OUTPUT /
    "06_ERROR_ANALYSIS" /
    "all_wrong_predictions.csv",

    index=False

)


# ============================================================
# 15. CORRECT CSV
# ============================================================

correct.to_csv(

    OUTPUT /
    "06_ERROR_ANALYSIS" /
    "all_correct_predictions.csv",

    index=False

)


# ============================================================
# 16. CROSS-SOURCE RESULTS
# ============================================================

cross_root = (
    ROOT /
    "CROSS_SOURCE_RESULTS"
)


if cross_root.exists():

    print(
        "\nProcessing cross-source results..."
    )


    for source_dir in cross_root.iterdir():

        if not source_dir.is_dir():

            continue


        target_dir = (

            OUTPUT /
            "13_CROSS_SOURCE" /
            source_dir.name

        )


        target_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        for file in source_dir.iterdir():

            if file.is_file():

                try:

                    shutil.copy2(

                        file,

                        target_dir /
                        file.name

                    )

                except:

                    pass


# ============================================================
# 17. PUBLICATION FIGURES
# ============================================================

pub_dir = (
    OUTPUT /
    "14_PUBLICATION_FIGURES"
)


pub_dir.mkdir(
    exist_ok=True
)


# Copy important figures
important_figures = [

    OUTPUT /
    "01_TRAINING" /
    "training_validation_accuracy.png",

    OUTPUT /
    "01_TRAINING" /
    "training_validation_loss.png",

    OUTPUT /
    "03_CONFUSION_MATRICES" /
    "confusion_matrix.png",

    OUTPUT /
    "03_CONFUSION_MATRICES" /
    "normalized_confusion_matrix.png",

    OUTPUT /
    "04_CLASS_METRICS" /
    "per_class_precision_recall_f1.png",

    OUTPUT /
    "06_ERROR_ANALYSIS" /
    "error_rate_by_class.png",

    OUTPUT /
    "06_ERROR_ANALYSIS" /
    "top_confusion_pairs.png"

]


for source in important_figures:

    if source.exists():

        shutil.copy2(

            source,

            pub_dir /
            source.name

        )


# ============================================================
# 18. FINAL MASTER REPORT
# ============================================================

report_txt = (

    OUTPUT /
    "15_REPORT" /
    "BOTTLE_GOURD_MASTER_REPORT.txt"

)


with open(

    report_txt,

    "w",

    encoding="utf-8"

) as f:


    f.write(
        "BOTTLE GOURD AI DISEASE DETECTION\n"
    )

    f.write(
        "COMPLETE MODEL ANALYSIS REPORT\n"
    )

    f.write(
        "=" * 80 +
        "\n\n"
    )


    f.write(
        "1. TEST SET SUMMARY\n"
    )

    f.write(
        "-" * 80 +
        "\n"
    )


    f.write(
        f"Test images: "
        f"{len(pred):,}\n"
    )


    f.write(
        f"Correct predictions: "
        f"{pred['_correct'].sum():,}\n"
    )


    f.write(
        f"Incorrect predictions: "
        f"{(~pred['_correct']).sum():,}\n"
    )


    f.write(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%\n"
    )


    f.write(
        f"Macro Precision: "
        f"{summary['macro_precision'] * 100:.2f}%\n"
    )


    f.write(
        f"Macro Recall: "
        f"{summary['macro_recall'] * 100:.2f}%\n"
    )


    f.write(
        f"Macro F1: "
        f"{summary['macro_f1'] * 100:.2f}%\n"
    )


    f.write(
        f"Weighted F1: "
        f"{summary['weighted_f1'] * 100:.2f}%\n\n"
    )


    f.write(
        "2. PER-CLASS RESULTS\n"
    )

    f.write(
        "-" * 80 +
        "\n"
    )


    for cls in classes:

        row = class_report.loc[
            cls
        ]


        f.write(
            f"\n{cls}\n"
        )


        f.write(
            f"  Precision: "
            f"{row['precision']:.4f}\n"
        )


        f.write(
            f"  Recall: "
            f"{row['recall']:.4f}\n"
        )


        f.write(
            f"  F1: "
            f"{row['f1-score']:.4f}\n"
        )


        f.write(
            f"  Support: "
            f"{int(row['support'])}\n"
        )


    f.write(
        "\n3. ERROR ANALYSIS\n"
    )

    f.write(
        "-" * 80 +
        "\n"
    )


    f.write(
        f"Total errors: "
        f"{len(wrong)}\n"
    )


    f.write(
        f"Error rate: "
        f"{len(wrong)/len(pred)*100:.2f}%\n"
    )


    if len(high_wrong) > 0:

        f.write(
            f"High-confidence errors (>=80%): "
            f"{len(high_wrong)}\n"
        )


    if len(low_wrong) > 0:

        f.write(
            f"Low-confidence errors (<50%): "
            f"{len(low_wrong)}\n"
        )


    f.write(
        "\nTop confusion pairs:\n"
    )


    for (
        (true_cls, pred_cls),
        count
    ) in pairs[:20]:

        f.write(

            f"  {true_cls} -> "
            f"{pred_cls}: "
            f"{count}\n"

        )


    f.write(
        "\n4. FILES GENERATED\n"
    )

    f.write(
        "-" * 80 +
        "\n"
    )


    f.write(
        "Training curves\n"
    )

    f.write(
        "Confusion matrices\n"
    )

    f.write(
        "Per-class metrics\n"
    )

    f.write(
        "Confidence analysis\n"
    )

    f.write(
        "Error analysis\n"
    )

    f.write(
        "ROC/PR curves where probability data exists\n"
    )

    f.write(
        "Correct prediction galleries\n"
    )

    f.write(
        "Wrong prediction galleries\n"
    )

    f.write(
        "High-confidence error gallery\n"
    )

    f.write(
        "Low-confidence error gallery\n"
    )

    f.write(
        "Per-class galleries\n"
    )

    f.write(
        "Cross-source results\n"
    )

    f.write(
        "Publication figures\n"
    )


# ============================================================
# 19. FINAL CONSOLE
# ============================================================

print("\n")

print("=" * 80)

print(
    "BOTTLE GOURD COMPLETE ANALYSIS FINISHED"
)

print("=" * 80)


print(
    f"\nTest images: "
    f"{len(pred):,}"
)


print(
    f"Correct: "
    f"{pred['_correct'].sum():,}"
)


print(
    f"Wrong: "
    f"{(~pred['_correct']).sum():,}"
)


print(
    f"Accuracy: "
    f"{accuracy * 100:.2f}%"
)


print(
    f"Macro F1: "
    f"{summary['macro_f1'] * 100:.2f}%"
)


print(
    f"\nHigh-confidence wrong: "
    f"{len(high_wrong)}"
)


print(
    f"Low-confidence wrong: "
    f"{len(low_wrong)}"
)


print(
    "\nOUTPUT:"
)

print(
    OUTPUT
)


print(
    "\nMASTER REPORT:"
)

print(
    report_txt
)


print(
    "\nPublication figures:"
)

print(
    pub_dir
)


print(
    "\nNo dataset/model files were modified."
)


print(
    "\nDONE."
)

input(
    "\nPress Enter to exit..."
)