import os
import json
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


# ============================================================
# STEP 19
# CROSS-SOURCE GENERALIZATION
#
# Experiment:
#   TRAIN = all sources except BG01_Mendeley
#   TEST  = BG01_Mendeley
#
# Images:
#   MASTER_DATASET through master_path
#
# Model:
#   EfficientNet-B0
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(
    r"D:\AI_Agriculture_Dataset\RAW_DATASETS\BOTTLE_GUARD_MASTER"
)

CROSS_SOURCE_DIR = (
    BASE_DIR / "CROSS_SOURCE_DATASET"
)

OUTPUT_DIR = (
    BASE_DIR /
    "CROSS_SOURCE_RESULTS" /
    "HOLDOUT_BG01_Mendeley"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. EXPERIMENT CONFIGURATION
# ============================================================

HELD_OUT_SOURCE = "BG01_Mendeley"

RANDOM_SEED = 42

IMAGE_SIZE = 224

BATCH_SIZE = 32

NUM_WORKERS = 0

EPOCHS = 15

LEARNING_RATE = 0.0001

WEIGHT_DECAY = 0.0001

VAL_SIZE = 0.15


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


set_seed(RANDOM_SEED)


# ============================================================
# 4. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("\n" + "=" * 70)
print("STEP 19: CROSS-SOURCE EFFICIENTNET TRAINING")
print("=" * 70)

print(
    f"\nHeld-out source : "
    f"{HELD_OUT_SOURCE}"
)

print(
    f"Device          : "
    f"{device}"
)


if torch.cuda.is_available():

    print(
        f"GPU             : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA version    : "
        f"{torch.version.cuda}"
    )


# ============================================================
# 5. INPUT FILES
# ============================================================

FOLD_DIR = (
    CROSS_SOURCE_DIR /
    f"HOLDOUT_{HELD_OUT_SOURCE}"
)

TRAIN_POOL_FILE = (
    FOLD_DIR /
    "train_pool.csv"
)

TEST_FILE = (
    FOLD_DIR /
    "test.csv"
)


if not TRAIN_POOL_FILE.exists():

    raise FileNotFoundError(
        f"\nTraining metadata not found:\n"
        f"{TRAIN_POOL_FILE}"
    )


if not TEST_FILE.exists():

    raise FileNotFoundError(
        f"\nTest metadata not found:\n"
        f"{TEST_FILE}"
    )


# ============================================================
# 6. LOAD METADATA
# ============================================================

print(
    "\nLoading cross-source metadata..."
)

train_pool = pd.read_csv(
    TRAIN_POOL_FILE
)

test_df = pd.read_csv(
    TEST_FILE
)


print(
    f"Training pool : "
    f"{len(train_pool):,}"
)

print(
    f"Held-out test  : "
    f"{len(test_df):,}"
)


# ============================================================
# 7. COMMON CLASSES
# ============================================================

classes = sorted(
    test_df[
        "standardized_class"
    ].unique()
)


train_pool = train_pool[
    train_pool[
        "standardized_class"
    ].isin(classes)
].copy()


test_df = test_df[
    test_df[
        "standardized_class"
    ].isin(classes)
].copy()


class_to_idx = {
    cls: idx
    for idx, cls in enumerate(classes)
}


idx_to_class = {
    idx: cls
    for cls, idx in class_to_idx.items()
}


print(
    f"\nNumber of classes: "
    f"{len(classes)}"
)

print("\nClasses:")

for idx, cls in idx_to_class.items():

    print(
        f"  {idx}: {cls}"
    )


# ============================================================
# 8. TRAIN / VALIDATION SPLIT
# ============================================================

print(
    "\nCreating training/validation split..."
)


train_labels = train_pool[
    "standardized_class"
].map(class_to_idx)


train_df, val_df = train_test_split(
    train_pool,
    test_size=VAL_SIZE,
    random_state=RANDOM_SEED,
    stratify=train_labels
)


train_df = train_df.reset_index(
    drop=True
)

val_df = val_df.reset_index(
    drop=True
)


test_df = test_df.reset_index(
    drop=True
)


print(
    f"Training images   : "
    f"{len(train_df):,}"
)

print(
    f"Validation images : "
    f"{len(val_df):,}"
)

print(
    f"Final test images : "
    f"{len(test_df):,}"
)


# ============================================================
# 9. VERIFY MASTER IMAGE PATHS
# ============================================================

print(
    "\nChecking master image paths..."
)


def check_master_paths(
    dataframe,
    name
):

    missing = []

    for path in dataframe[
        "master_path"
    ]:

        path = Path(
            str(path)
        )

        if not path.is_absolute():

            path = BASE_DIR / path

        if not path.exists():

            missing.append(
                str(path)
            )


    if missing:

        print(
            f"\nERROR: "
            f"{len(missing)} missing "
            f"images in {name}"
        )

        print(
            "\nFirst missing paths:"
        )

        for path in missing[:10]:

            print(path)

        raise FileNotFoundError(
            f"\nMissing master images "
            f"in {name}."
        )


    print(
        f"{name}: all "
        f"{len(dataframe):,} "
        f"image paths valid."
    )


check_master_paths(
    train_df,
    "TRAIN"
)

check_master_paths(
    val_df,
    "VALIDATION"
)

check_master_paths(
    test_df,
    "TEST"
)


# ============================================================
# 10. DATASET
# ============================================================

class BottleGourdDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        class_to_idx,
        transform=None
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.class_to_idx = class_to_idx

        self.transform = transform


    def __len__(self):

        return len(self.df)


    def __getitem__(self, index):

        row = self.df.iloc[index]


        image_path = Path(
            str(row["master_path"])
        )


        if not image_path.is_absolute():

            image_path = (
                BASE_DIR /
                image_path
            )


        image = Image.open(
            image_path
        ).convert("RGB")


        label = self.class_to_idx[
            row["standardized_class"]
        ]


        if self.transform:

            image = self.transform(
                image
            )


        return image, label


# ============================================================
# 11. TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=15
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


eval_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# 12. CREATE DATASETS
# ============================================================

train_dataset = BottleGourdDataset(
    train_df,
    class_to_idx,
    train_transform
)


val_dataset = BottleGourdDataset(
    val_df,
    class_to_idx,
    eval_transform
)


test_dataset = BottleGourdDataset(
    test_df,
    class_to_idx,
    eval_transform
)


# ============================================================
# 13. DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# 14. CLASS WEIGHTS
# ============================================================

train_class_counts = np.zeros(
    len(classes),
    dtype=np.float64
)


for cls in classes:

    idx = class_to_idx[cls]

    train_class_counts[idx] = np.sum(
        train_df[
            "standardized_class"
        ].values == cls
    )


total_train = np.sum(
    train_class_counts
)


num_classes = len(classes)


class_weights = (
    total_train /
    (
        num_classes *
        train_class_counts
    )
)


class_weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=device
)


print(
    "\nTraining class distribution:"
)


for idx, cls in idx_to_class.items():

    print(
        f"  {cls:<35}"
        f"{int(train_class_counts[idx]):>6}"
        f"  weight={class_weights[idx]:.4f}"
    )


# ============================================================
# 15. MODEL
# ============================================================

print(
    "\nLoading pretrained EfficientNet-B0..."
)


weights = (
    models.EfficientNet_B0_Weights.DEFAULT
)


model = models.efficientnet_b0(
    weights=weights
)


in_features = (
    model.classifier[1].in_features
)


model.classifier[1] = nn.Linear(
    in_features,
    len(classes)
)


model = model.to(device)


# ============================================================
# 16. LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor
)


# ============================================================
# 17. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 18. LR SCHEDULER
# ============================================================

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )
)


# ============================================================
# 19. AMP
# ============================================================

use_amp = torch.cuda.is_available()


if use_amp:

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


# ============================================================
# 20. TRAIN FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        if use_amp:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    labels
                )


            scaler.scale(
                loss
            ).backward()


            scaler.step(
                optimizer
            )


            scaler.update()


        else:

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()


        running_loss += (
            loss.item() *
            images.size(0)
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


    epoch_loss = (
        running_loss /
        total
    )


    epoch_acc = (
        correct /
        total
    )


    return epoch_loss, epoch_acc


# ============================================================
# 21. EVALUATION FUNCTION
# ============================================================

def evaluate_loader(
    loader
):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0


    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )


            if use_amp:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):

                    outputs = model(
                        images
                    )

                    loss = criterion(
                        outputs,
                        labels
                    )

            else:

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    labels
                )


            running_loss += (
                loss.item() *
                images.size(0)
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += labels.size(0)


    loss_value = (
        running_loss /
        total
    )


    accuracy = (
        correct /
        total
    )


    return loss_value, accuracy


# ============================================================
# 22. TRAINING LOOP
# ============================================================

history = []


best_val_accuracy = -1.0


best_epoch = 0


best_model_path = (
    OUTPUT_DIR /
    "best_cross_source_efficientnet_b0.pth"
)


print(
    "\n" + "=" * 70
)

print(
    "STARTING TRAINING"
)

print(
    "=" * 70
)


for epoch in range(
    1,
    EPOCHS + 1
):

    start_time = time.time()


    train_loss, train_acc = (
        train_one_epoch()
    )


    val_loss, val_acc = (
        evaluate_loader(
            val_loader
        )
    )


    scheduler.step(
        val_acc
    )


    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    elapsed = (
        time.time() -
        start_time
    )


    history.append({

        "epoch":
            epoch,

        "train_loss":
            train_loss,

        "train_accuracy":
            train_acc,

        "val_loss":
            val_loss,

        "val_accuracy":
            val_acc,

        "learning_rate":
            current_lr,

        "epoch_time_seconds":
            elapsed

    })


    print(
        f"\nEpoch "
        f"{epoch:02d}/{EPOCHS}"
    )


    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )


    print(
        f"Train Acc : "
        f"{train_acc * 100:.2f}%"
    )


    print(
        f"Val Loss  : "
        f"{val_loss:.4f}"
    )


    print(
        f"Val Acc   : "
        f"{val_acc * 100:.2f}%"
    )


    print(
        f"LR        : "
        f"{current_lr:.8f}"
    )


    print(
        f"Time      : "
        f"{elapsed:.1f} sec"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_acc > best_val_accuracy:

        best_val_accuracy = val_acc

        best_epoch = epoch


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "classes":
                    classes,

                "class_to_idx":
                    class_to_idx,

                "epoch":
                    epoch,

                "val_accuracy":
                    val_acc,

                "held_out_source":
                    HELD_OUT_SOURCE,

                "random_seed":
                    RANDOM_SEED
            },
            best_model_path
        )


        print(
            ">>> BEST MODEL SAVED"
        )


# ============================================================
# 23. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)


history_df.to_csv(
    OUTPUT_DIR /
    "training_history.csv",
    index=False
)


# ============================================================
# 24. LOAD BEST MODEL
# ============================================================

print(
    "\nLoading best validation model..."
)


checkpoint = torch.load(
    best_model_path,
    map_location=device
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


print(
    f"Best epoch      : "
    f"{checkpoint['epoch']}"
)


print(
    f"Best validation : "
    f"{checkpoint['val_accuracy'] * 100:.2f}%"
)


# ============================================================
# 25. HELD-OUT TEST
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "FINAL HELD-OUT SOURCE TEST"
)

print(
    "=" * 70
)


model.eval()


all_true = []

all_pred = []

all_probabilities = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )


        if use_amp:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = model(
                    images
                )

        else:

            outputs = model(
                images
            )


        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        predictions = torch.argmax(
            probabilities,
            dim=1
        )


        all_true.extend(
            labels.cpu().numpy()
        )


        all_pred.extend(
            predictions.cpu().numpy()
        )


        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


all_true = np.array(
    all_true
)

all_pred = np.array(
    all_pred
)

all_probabilities = np.array(
    all_probabilities
)


# ============================================================
# 26. METRICS
# ============================================================

test_accuracy = accuracy_score(
    all_true,
    all_pred
)


(
    precision_macro,
    recall_macro,
    f1_macro,
    _
) = precision_recall_fscore_support(
    all_true,
    all_pred,
    average="macro",
    zero_division=0
)


(
    precision_weighted,
    recall_weighted,
    f1_weighted,
    _
) = precision_recall_fscore_support(
    all_true,
    all_pred,
    average="weighted",
    zero_division=0
)


print(
    f"\nHeld-out source:"
    f" {HELD_OUT_SOURCE}"
)


print(
    f"Test images:"
    f" {len(all_true):,}"
)


print(
    f"\nTest Accuracy       : "
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
    f"Macro F1            : "
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
    f"Weighted F1         : "
    f"{f1_weighted * 100:.2f}%"
)


# ============================================================
# 27. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_true,
    all_pred,
    labels=list(
        range(len(classes))
    ),
    target_names=classes,
    output_dict=True,
    zero_division=0
)


report_df = pd.DataFrame(
    report
).transpose()


report_df.to_csv(
    OUTPUT_DIR /
    "classification_report.csv"
)


# ============================================================
# 28. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_true,
    all_pred,
    labels=list(
        range(len(classes))
    )
)


cm_df = pd.DataFrame(
    cm,
    index=classes,
    columns=classes
)


cm_df.to_csv(
    OUTPUT_DIR /
    "confusion_matrix.csv"
)


# ============================================================
# 29. TEST PREDICTIONS
# ============================================================

prediction_df = test_df.copy()


prediction_df[
    "true_class_index"
] = all_true


prediction_df[
    "predicted_class_index"
] = all_pred


prediction_df[
    "predicted_class"
] = [
    idx_to_class[x]
    for x in all_pred
]


prediction_df[
    "confidence"
] = np.max(
    all_probabilities,
    axis=1
)


prediction_df[
    "correct"
] = (
    all_true ==
    all_pred
)


prediction_df.to_csv(
    OUTPUT_DIR /
    "test_predictions.csv",
    index=False
)


# ============================================================
# 30. SUMMARY
# ============================================================

summary = {

    "experiment":
        "Leave-One-Source-Out Cross-Dataset Generalization",

    "held_out_source":
        HELD_OUT_SOURCE,

    "train_pool_images":
        int(len(train_pool)),

    "train_images":
        int(len(train_df)),

    "validation_images":
        int(len(val_df)),

    "test_images":
        int(len(test_df)),

    "number_of_classes":
        int(len(classes)),

    "classes":
        classes,

    "best_epoch":
        int(best_epoch),

    "best_validation_accuracy":
        float(best_val_accuracy),

    "test_accuracy":
        float(test_accuracy),

    "macro_precision":
        float(precision_macro),

    "macro_recall":
        float(recall_macro),

    "macro_f1":
        float(f1_macro),

    "weighted_precision":
        float(precision_weighted),

    "weighted_recall":
        float(recall_weighted),

    "weighted_f1":
        float(f1_weighted),

    "random_seed":
        RANDOM_SEED,

    "batch_size":
        BATCH_SIZE,

    "epochs":
        EPOCHS,

    "learning_rate":
        LEARNING_RATE,

    "weight_decay":
        WEIGHT_DECAY,

    "model":
        "EfficientNet-B0",

    "image_size":
        IMAGE_SIZE

}


with open(
    OUTPUT_DIR /
    "cross_source_summary.json",
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# 31. FINAL
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STEP 19 COMPLETED SUCCESSFULLY"
)

print(
    "=" * 70
)


print(
    f"\nHeld-out source:"
    f" {HELD_OUT_SOURCE}"
)


print(
    f"Best validation accuracy:"
    f" {best_val_accuracy * 100:.2f}%"
)


print(
    f"Held-out test accuracy:"
    f" {test_accuracy * 100:.2f}%"
)


print(
    f"Held-out macro F1:"
    f" {f1_macro * 100:.2f}%"
)


print(
    f"\nResults saved to:"
)


print(
    OUTPUT_DIR
)


print("\nGenerated files:")

print(
    "  best_cross_source_efficientnet_b0.pth"
)

print(
    "  training_history.csv"
)

print(
    "  classification_report.csv"
)

print(
    "  confusion_matrix.csv"
)

print(
    "  test_predictions.csv"
)

print(
    "  cross_source_summary.json"
)


print(
    "\n" + "=" * 70
)