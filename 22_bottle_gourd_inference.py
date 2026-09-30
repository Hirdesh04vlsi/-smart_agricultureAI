import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    r"D:\AI_Agriculture_Dataset\RAW_DATASETS\BOTTLE_GUARD_MASTER"
)

MODEL_PATH = BASE_DIR / "TRAINING_OUTPUT" / "best_efficientnet_b0.pth"
MODEL_INFO_PATH = BASE_DIR / "TRAINING_OUTPUT" / "model_info.json"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("BOTTLE GOURD DISEASE INFERENCE")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU   : {torch.cuda.get_device_name(0)}")


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Alternaria_Leaf_Blight",
    "Anthracnose",
    "Downy_Mildew",
    "Dry_Leaf",
    "Early_Alternaria_Leaf_Blight",
    "Fungal_Damage_Leaf",
    "Healthy",
    "Mosaic_Virus",
    "Nutrition_Deficiency",
    "Pest_Infestation",
]


# ============================================================
# MODEL
# ============================================================

print("\nLoading EfficientNet-B0...")

model = models.efficientnet_b0(weights=None)

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    len(CLASS_NAMES)
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

# Handle either a raw state_dict or a checkpoint dictionary
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image_path, top_k=3):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        values, indices = torch.topk(
            probabilities,
            k=min(top_k, len(CLASS_NAMES)),
            dim=1
        )

    results = []

    for probability, index in zip(
        values[0],
        indices[0]
    ):

        results.append({
            "class": CLASS_NAMES[index.item()],
            "confidence": float(probability.item())
        })

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    image_input = input(
        "\nEnter image path: "
    ).strip().strip('"')

    try:

        predictions = predict_image(
            image_input,
            top_k=3
        )

        print("\n" + "=" * 70)
        print("PREDICTION")
        print("=" * 70)

        for i, result in enumerate(
            predictions,
            start=1
        ):

            print(
                f"{i}. "
                f"{result['class']:<35} "
                f"{result['confidence'] * 100:.2f}%"
            )

        print("=" * 70)

    except Exception as e:

        print("\nERROR:")
        print(e)