from pathlib import Path
import json

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from flask import (
    Flask,
    render_template,
    request,
    url_for
)

from werkzeug.utils import secure_filename


# ============================================================
# PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    APP_DIR
    / "models"
    / "bottle_gourd"
    / "model.pth"
)

RECOMMENDATION_PATH = (
    APP_DIR
    / "recommendations"
    / "bottle_gourd.json"
)

UPLOAD_DIR = APP_DIR / "uploads"

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("SMART AGRICULTURE AI")
print("BOTTLE GOURD DISEASE DETECTION")
print("=" * 70)

print(
    f"Device: {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


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
# LOAD RECOMMENDATIONS
# ============================================================

print(
    "\nLoading recommendation database..."
)


with open(
    RECOMMENDATION_PATH,
    "r",
    encoding="utf-8"
) as f:

    RECOMMENDATIONS = json.load(f)


print(
    f"Crop: {RECOMMENDATIONS['crop']['name']}"
)

print(
    f"Classes: {len(RECOMMENDATIONS['diseases'])}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "\nLoading EfficientNet-B0..."
)


model = models.efficientnet_b0(
    weights=None
)


num_features = (
    model.classifier[1].in_features
)


model.classifier[1] = nn.Linear(
    num_features,
    len(CLASS_NAMES)
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)


if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    state_dict = checkpoint[
        "model_state_dict"
    ]

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict
)


model.to(DEVICE)

model.eval()


print(
    "Model loaded successfully."
)


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
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
# PREDICTION
# ============================================================

def predict_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")


    tensor = transform(
        image
    )


    tensor = tensor.unsqueeze(
        0
    )


    tensor = tensor.to(
        DEVICE
    )


    with torch.no_grad():

        outputs = model(
            tensor
        )


        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        values, indices = torch.topk(

            probabilities,

            k=3,

            dim=1

        )


    predictions = []


    for probability, index in zip(

        values[0],

        indices[0]

    ):

        class_name = CLASS_NAMES[
            index.item()
        ]


        info = RECOMMENDATIONS[
            "diseases"
        ].get(

            class_name,

            {}

        )


        predictions.append({

            "class": class_name,

            "confidence":
                float(
                    probability.item()
                ),

            "info": info

        })


    return predictions


# ============================================================
# FILE VALIDATION
# ============================================================

def allowed_file(filename):

    extension = Path(
        filename
    ).suffix.lower()


    return extension in ALLOWED_EXTENSIONS


# ============================================================
# HOME
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)

def home():

    return render_template(

        "index.html",

        predictions=None,

        crop=RECOMMENDATIONS[
            "crop"
        ]

    )


# ============================================================
# PREDICT
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)

def predict():


    # --------------------------------------------------------
    # IMPORTANT:
    # HTML uses name="file"
    # --------------------------------------------------------

    if "file" not in request.files:

        return render_template(

            "index.html",

            predictions=None,

            error="No image selected.",

            crop=RECOMMENDATIONS[
                "crop"
            ]

        )


    file = request.files[
        "file"
    ]


    # --------------------------------------------------------
    # EMPTY FILE
    # --------------------------------------------------------

    if file.filename == "":

        return render_template(

            "index.html",

            predictions=None,

            error="Please select an image.",

            crop=RECOMMENDATIONS[
                "crop"
            ]

        )


    # --------------------------------------------------------
    # EXTENSION CHECK
    # --------------------------------------------------------

    if not allowed_file(
        file.filename
    ):

        return render_template(

            "index.html",

            predictions=None,

            error="Unsupported image format.",

            crop=RECOMMENDATIONS[
                "crop"
            ]

        )


    # --------------------------------------------------------
    # SECURE FILENAME
    # --------------------------------------------------------

    filename = secure_filename(
        file.filename
    )


    # --------------------------------------------------------
    # AVOID OLD FILE CONFLICTS
    # --------------------------------------------------------

    save_path = (
        UPLOAD_DIR
        / filename
    )


    # --------------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------------

    file.save(
        save_path
    )


    print(
        f"\nImage received: {filename}"
    )

    print(
        f"Saved to: {save_path}"
    )


    try:


        # ----------------------------------------------------
        # RUN MODEL
        # ----------------------------------------------------

        predictions = predict_image(
            save_path
        )


        # ----------------------------------------------------
        # CREATE BROWSER URL
        # ----------------------------------------------------

        image_url = url_for(
            "uploaded_file",
            filename=filename
        )


        # ----------------------------------------------------
        # PRINT RESULT
        # ----------------------------------------------------

        print(
            "\nPrediction:"
        )

        for prediction in predictions:

            print(

                f"  "
                f"{prediction['class']} : "
                f"{prediction['confidence'] * 100:.2f}%"

            )


        # ----------------------------------------------------
        # RETURN RESULT PAGE
        # ----------------------------------------------------

        return render_template(

            "index.html",

            predictions=predictions,

            crop=RECOMMENDATIONS[
                "crop"
            ],

            image_url=image_url

        )


    except Exception as e:


        print(
            f"\nPrediction error: {e}"
        )


        return render_template(

            "index.html",

            predictions=None,

            error=str(e),

            crop=RECOMMENDATIONS[
                "crop"
            ]

        )


# ============================================================
# SERVE UPLOADED IMAGES
# ============================================================

@app.route(
    "/uploads/<filename>"
)

def uploaded_file(
    filename
):

    from flask import send_from_directory


    return send_from_directory(

        UPLOAD_DIR,

        filename

    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":


    print()

    print(
        "=" * 70
    )

    print(
        "SERVER READY"
    )

    print(
        "=" * 70
    )

    print(
        "Open: http://127.0.0.1:5000"
    )

    print(
        "=" * 70
    )


    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False

    )