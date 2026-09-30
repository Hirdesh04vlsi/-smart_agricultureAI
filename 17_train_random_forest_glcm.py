import os
import cv2
import joblib
import numpy as np
import pandas as pd

from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\AI_Agriculture_Dataset\RAW_DATASETS\BOTTLE_GUARD_MASTER"

DATASET_DIR = os.path.join(
    BASE_DIR,
    "SPLIT_DATASET"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "RF_GLCM_RESULTS"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)

RANDOM_STATE = 42

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    image = cv2.resize(
        image,
        IMAGE_SIZE
    )

    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # COLOR FEATURES
    # --------------------------------------------------------

    color_features = []

    for channel in range(3):

        channel_data = rgb[:, :, channel]

        color_features.extend([
            np.mean(channel_data),
            np.std(channel_data),
            np.min(channel_data),
            np.max(channel_data)
        ])

    # --------------------------------------------------------
    # HSV FEATURES
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    for channel in range(3):

        channel_data = hsv[:, :, channel]

        color_features.extend([
            np.mean(channel_data),
            np.std(channel_data),
            np.min(channel_data),
            np.max(channel_data)
        ])

    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Quantize grayscale to 32 levels
    gray_quantized = (
        gray / 8
    ).astype(np.uint8)

    # --------------------------------------------------------
    # GLCM
    # --------------------------------------------------------

    glcm = graycomatrix(
        gray_quantized,
        distances=[1, 2],
        angles=[
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4
        ],
        levels=32,
        symmetric=True,
        normed=True
    )

    texture_features = []

    properties = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM"
    ]

    for prop in properties:

        values = graycoprops(
            glcm,
            prop
        )

        texture_features.extend([
            np.mean(values),
            np.std(values)
        ])

    # --------------------------------------------------------
    # SHAPE / EDGE FEATURES
    # --------------------------------------------------------

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_density = np.mean(
        edges > 0
    )

    shape_features = [
        edge_density,
        np.mean(gray),
        np.std(gray)
    ]

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    features = (
        color_features
        + texture_features
        + shape_features
    )

    return np.array(
        features,
        dtype=np.float32
    )


# ============================================================
# DATASET FEATURE EXTRACTION
# ============================================================

def process_split(split_name):

    split_dir = os.path.join(
        DATASET_DIR,
        split_name
    )

    X = []
    y = []
    paths = []

    class_names = sorted([
        d for d in os.listdir(split_dir)
        if os.path.isdir(
            os.path.join(
                split_dir,
                d
            )
        )
    ])

    print("\n" + "=" * 70)
    print(
        f"PROCESSING {split_name.upper()} SET"
    )
    print("=" * 70)

    print(
        f"Classes: {len(class_names)}"
    )

    for class_name in class_names:

        class_dir = os.path.join(
            split_dir,
            class_name
        )

        image_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp"
                )
            )
        ]

        print(
            f"{class_name}: "
            f"{len(image_files)} images"
        )

        for filename in tqdm(
            image_files,
            desc=class_name
        ):

            image_path = os.path.join(
                class_dir,
                filename
            )

            try:

                features = extract_features(
                    image_path
                )

                X.append(features)
                y.append(class_name)
                paths.append(image_path)

            except Exception as e:

                print(
                    f"\nERROR: {image_path}"
                )

                print(e)

    X = np.array(X)

    y = np.array(y)

    print(
        f"\nFeatures shape: {X.shape}"
    )

    return (
        X,
        y,
        paths,
        class_names
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("BOTTLE GOURD - RANDOM FOREST + GLCM BASELINE")
print("=" * 70)

print(
    "\nThis experiment uses the SAME train/test split "
    "as the EfficientNet experiment."
)

# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

X_train, y_train, train_paths, class_names = process_split(
    "train"
)

# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

X_val, y_val, val_paths, _ = process_split(
    "val"
)

# ------------------------------------------------------------
# TEST
# ------------------------------------------------------------

X_test, y_test, test_paths, _ = process_split(
    "test"
)

# ============================================================
# SAVE EXTRACTED FEATURES
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_train.npy"
    ),
    X_train
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_val.npy"
    ),
    X_val
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_test.npy"
    ),
    X_test
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_train.npy"
    ),
    y_train
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_val.npy"
    ),
    y_val
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_test.npy"
    ),
    y_test
)

# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

model = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)

print(
    "\nRandom Forest training complete."
)

# ============================================================
# VALIDATION
# ============================================================

val_pred = model.predict(
    X_val
)

val_accuracy = accuracy_score(
    y_val,
    val_pred
)

print(
    f"\nValidation Accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

# ============================================================
# TEST
# ============================================================

test_pred = model.predict(
    X_test
)

test_accuracy = accuracy_score(
    y_test,
    test_pred
)

precision_macro, recall_macro, f1_macro, _ = (
    precision_recall_fscore_support(
        y_test,
        test_pred,
        average="macro",
        zero_division=0
    )
)

precision_weighted, recall_weighted, f1_weighted, _ = (
    precision_recall_fscore_support(
        y_test,
        test_pred,
        average="weighted",
        zero_division=0
    )
)

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST TEST RESULTS")
print("=" * 70)

print(
    f"Test Accuracy       : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Macro Precision     : "
    f"{precision_macro * 100:.2f}%"
)

print(
    f"Macro Recall        : "
    f"{recall_macro * 100:.2f}%"
)

print(
    f"Macro F1-score      : "
    f"{f1_macro * 100:.2f}%"
)

print(
    f"Weighted Precision  : "
    f"{precision_weighted * 100:.2f}%"
)

print(
    f"Weighted Recall     : "
    f"{recall_weighted * 100:.2f}%"
)

print(
    f"Weighted F1-score   : "
    f"{f1_weighted * 100:.2f}%"
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    test_pred,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report
).transpose()

report_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "classification_report.csv"
    )
)

print("\n" + "=" * 70)
print("PER-CLASS TEST RESULTS")
print("=" * 70)

print(
    report_df.to_string()
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_pred,
    labels=class_names
)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

cm_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.csv"
    )
)

# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    os.path.join(
        OUTPUT_DIR,
        "random_forest_glcm.pkl"
    )
)

# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame([{
    "Model": "Random Forest + Color + GLCM + Edge Features",
    "Train_Images": len(y_train),
    "Validation_Images": len(y_val),
    "Test_Images": len(y_test),
    "Feature_Count": X_train.shape[1],
    "Validation_Accuracy": val_accuracy,
    "Test_Accuracy": test_accuracy,
    "Macro_Precision": precision_macro,
    "Macro_Recall": recall_macro,
    "Macro_F1": f1_macro,
    "Weighted_Precision": precision_weighted,
    "Weighted_Recall": recall_weighted,
    "Weighted_F1": f1_weighted
}])

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rf_glcm_summary.csv"
    ),
    index=False
)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    f"Results folder:\n{OUTPUT_DIR}"
)

print(
    "\nRANDOM FOREST + GLCM EXPERIMENT COMPLETE."
)

print("=" * 70)