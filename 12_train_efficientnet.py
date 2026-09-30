import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

DATASET_ROOT = ROOT / "RESIZED_DATASET"
OUTPUT_ROOT = ROOT / "TRAINING_OUTPUT"

TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 15

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

SEED = 42


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    USE_AMP = DEVICE.type == "cuda"

    print("=" * 70)
    print("BOTTLE GOURD DISEASE CLASSIFICATION")
    print("EfficientNet-B0 Transfer Learning")
    print("=" * 70)

    print("\nDevice:", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "GPU memory:",
            round(
                torch.cuda.get_device_properties(0)
                .total_memory / (1024 ** 3),
                2
            ),
            "GB"
        )

    # --------------------------------------------------------
    # CUDA optimization
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.backends.cudnn.benchmark = True

    # ========================================================
    # TRANSFORMS
    # ========================================================

    # Images are already 224 x 224.
    # Therefore NO Resize() is performed every epoch.

    train_transform = transforms.Compose([

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            15
        ),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
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


    val_transform = transforms.Compose([

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

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        VAL_DIR,
        transform=val_transform
    )

    class_names = train_dataset.classes

    num_classes = len(class_names)

    print(
        "\nClasses:",
        num_classes
    )

    for i, name in enumerate(class_names):

        print(
            f"{i}: {name}"
        )

    print(
        "\nImages:",
        f"train={len(train_dataset)},",
        f"validation={len(val_dataset)}"
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=True,

        persistent_workers=False
    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=True,

        persistent_workers=False
    )

    print(
        "\nBatches per epoch:",
        f"train={len(train_loader)},",
        f"validation={len(val_loader)}"
    )

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    class_counts = torch.zeros(
        num_classes
    )

    for _, label in train_dataset.samples:

        class_counts[label] += 1


    class_weights = (
        1.0 / class_counts
    )

    class_weights = (
        class_weights /
        class_weights.mean()
    )

    class_weights = class_weights.to(
        DEVICE
    )

    print("\nClass weights:")

    for i, weight in enumerate(
        class_weights
    ):

        print(
            f"{class_names[i]:35s}: "
            f"{weight.item():.4f}"
        )

    # ========================================================
    # MODEL
    # ========================================================

    print(
        "\nLoading EfficientNet-B0..."
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
        num_classes
    )

    model = model.to(
        DEVICE
    )

    print(
        "Model loaded successfully."
    )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )

    # ========================================================
    # LR SCHEDULER
    # ========================================================

    scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(

            optimizer,

            mode="min",

            factor=0.5,

            patience=2
        )
    )

    # ========================================================
    # MIXED PRECISION
    # ========================================================

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP
    )

    print(
        "\nMixed precision:",
        USE_AMP
    )

    # ========================================================
    # TRAIN FUNCTION
    # ========================================================

    def train_one_epoch():

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        for images, labels in train_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(

                device_type="cuda",

                dtype=torch.float16,

                enabled=USE_AMP
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

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

        epoch_loss = (
            running_loss / total
        )

        epoch_accuracy = (
            correct / total
        )

        return (
            epoch_loss,
            epoch_accuracy
        )

    # ========================================================
    # VALIDATION FUNCTION
    # ========================================================

    def validate():

        model.eval()

        running_loss = 0.0

        correct = 0

        total = 0

        with torch.inference_mode():

            for images, labels in val_loader:

                images = images.to(
                    DEVICE,
                    non_blocking=True
                )

                labels = labels.to(
                    DEVICE,
                    non_blocking=True
                )

                with torch.autocast(

                    device_type="cuda",

                    dtype=torch.float16,

                    enabled=USE_AMP
                ):

                    outputs = model(
                        images
                    )

                    loss = criterion(
                        outputs,
                        labels
                    )

                running_loss += (
                    loss.item()
                    * images.size(0)
                )

                predictions = (
                    outputs.argmax(
                        dim=1
                    )
                )

                total += labels.size(0)

                correct += (
                    predictions == labels
                ).sum().item()

        epoch_loss = (
            running_loss / total
        )

        epoch_accuracy = (
            correct / total
        )

        return (
            epoch_loss,
            epoch_accuracy
        )

    # ========================================================
    # TRAINING
    # ========================================================

    history = {

        "train_loss": [],

        "train_accuracy": [],

        "val_loss": [],

        "val_accuracy": []
    }

    best_val_accuracy = 0.0
    best_epoch = 0

    print("\n")

    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    start_time = time.time()

    for epoch in range(
        NUM_EPOCHS
    ):

        epoch_start = time.time()

        train_loss, train_accuracy = (
            train_one_epoch()
        )

        val_loss, val_accuracy = (
            validate()
        )

        scheduler.step(
            val_loss
        )

        history[
            "train_loss"
        ].append(
            train_loss
        )

        history[
            "train_accuracy"
        ].append(
            train_accuracy
        )

        history[
            "val_loss"
        ].append(
            val_loss
        )

        history[
            "val_accuracy"
        ].append(
            val_accuracy
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f} | "
            f"Train Acc: "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Val Loss: "
            f"{val_loss:.4f} | "
            f"Val Acc: "
            f"{val_accuracy * 100:.2f}%"
        )

        print(
            f"Learning Rate: "
            f"{current_lr:.7f}"
        )

        print(
            f"Epoch Time: "
            f"{epoch_time:.1f} sec"
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = (
                val_accuracy
            )

            best_epoch = (
                epoch + 1
            )

            checkpoint = {

                "epoch":
                    epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "class_names":
                    class_names,

                "num_classes":
                    num_classes,

                "image_size":
                    IMAGE_SIZE,

                "best_val_accuracy":
                    best_val_accuracy
            }

            torch.save(

                checkpoint,

                OUTPUT_ROOT /
                "best_efficientnet_b0.pth"
            )

            print(
                "*** BEST MODEL SAVED "
                f"({best_val_accuracy * 100:.2f}%) ***"
            )

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    with open(

        OUTPUT_ROOT /
        "training_history.json",

        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )

    # ========================================================
    # SAVE MODEL INFO
    # ========================================================

    model_info = {

        "model":
            "EfficientNet-B0",

        "classes":
            class_names,

        "num_classes":
            num_classes,

        "image_size":
            IMAGE_SIZE,

        "batch_size":
            BATCH_SIZE,

        "epochs":
            NUM_EPOCHS,

        "train_images":
            len(train_dataset),

        "validation_images":
            len(val_dataset),

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "best_validation_accuracy":
            best_val_accuracy,

        "best_epoch":
            best_epoch,

        "device":
            str(DEVICE),

        "dataset":
            "RESIZED_DATASET"
    }

    with open(

        OUTPUT_ROOT /
        "model_info.json",

        "w"
    ) as f:

        json.dump(
            model_info,
            f,
            indent=4
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    total_time = (
        time.time()
        - start_time
    )

    print("\n")

    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"\nBest Validation Accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Best Epoch: "
        f"{best_epoch}"
    )

    print(
        f"Total Training Time: "
        f"{total_time / 60:.2f} minutes"
    )

    print(
        "\nBest model saved at:"
    )

    print(
        OUTPUT_ROOT /
        "best_efficientnet_b0.pth"
    )


# ============================================================
# WINDOWS ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()