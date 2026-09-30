from pathlib import Path

from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    r"D:\AI_Agriculture_Dataset\RAW_DATASETS\BOTTLE_GUARD_MASTER"
)

MODEL_PATH = (
    BASE_DIR
    / "TRAINING_OUTPUT"
    / "best_efficientnet_b0.pth"
)

UPLOAD_DIR = BASE_DIR / "WEB_UPLOADS"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


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
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("BOTTLE GOURD DISEASE DETECTION WEB SERVER")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU   : {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# CLASSES
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
# FARMER-FRIENDLY INFORMATION
# ============================================================

DISEASE_INFO = {

    "Alternaria_Leaf_Blight": {
        "name": "Alternaria Leaf Blight",
        "hindi": "अल्टरनेरिया पत्ती झुलसा",
        "advice": "प्रभावित पत्तियों को अलग करें और खेत में हवा का अच्छा संचार रखें।"
    },

    "Anthracnose": {
        "name": "Anthracnose",
        "hindi": "एन्थ्रेक्नोज",
        "advice": "प्रभावित पत्तियों को हटाएँ और खेत में अत्यधिक नमी तथा पत्तियों पर लंबे समय तक पानी रुकने से बचाएँ।"
    },

    "Downy_Mildew": {
        "name": "Downy Mildew",
        "hindi": "डाउनी मिल्ड्यू",
        "advice": "पत्तियों को लंबे समय तक गीला रहने से बचाएँ और खेत में पर्याप्त वायु संचार रखें।"
    },

    "Dry_Leaf": {
        "name": "Dry Leaf",
        "hindi": "सूखी पत्ती",
        "advice": "पौधे की सिंचाई, मिट्टी की नमी और पोषण की स्थिति की जाँच करें।"
    },

    "Early_Alternaria_Leaf_Blight": {
        "name": "Early Alternaria Leaf Blight",
        "hindi": "प्रारंभिक अल्टरनेरिया पत्ती झुलसा",
        "advice": "प्रभावित पत्तियों को हटाएँ और पत्तियों पर अनावश्यक नमी से बचाएँ।"
    },

    "Fungal_Damage_Leaf": {
        "name": "Fungal Damage",
        "hindi": "फफूंद से पत्ती क्षति",
        "advice": "प्रभावित भागों को हटाएँ और खेत में नमी तथा वायु संचार का ध्यान रखें।"
    },

    "Healthy": {
        "name": "Healthy Leaf",
        "hindi": "स्वस्थ पत्ती",
        "advice": "पौधा स्वस्थ दिखाई दे रहा है। नियमित सिंचाई, पोषण और निगरानी जारी रखें।"
    },

    "Mosaic_Virus": {
        "name": "Mosaic Virus",
        "hindi": "मोज़ेक वायरस",
        "advice": "प्रभावित पौधों को अलग रखें और वायरस फैलाने वाले कीटों की नियमित निगरानी करें।"
    },

    "Nutrition_Deficiency": {
        "name": "Nutrition Deficiency",
        "hindi": "पोषक तत्वों की कमी",
        "advice": "मिट्टी की जाँच कराएँ और पोषक तत्वों की स्थिति के अनुसार संतुलित पोषण दें।"
    },

    "Pest_Infestation": {
        "name": "Pest Infestation",
        "hindi": "कीट प्रकोप",
        "advice": "पत्तियों और पौधे पर कीटों की जाँच करें तथा एकीकृत कीट प्रबंधन अपनाएँ।"
    },
}


# ============================================================
# MODEL
# ============================================================

print("\nLoading EfficientNet-B0...")

model = models.efficientnet_b0(
    weights=None
)

num_features = model.classifier[1].in_features

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
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

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

        predictions.append({
            "class": class_name,
            "confidence": float(
                probability.item()
            ),
            "info": DISEASE_INFO[
                class_name
            ]
        })

    return predictions


# ============================================================
# HTML
# ============================================================

HTML = """
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>
Bottle Gourd Disease Detection
</title>

<style>

body {
    font-family: Arial, sans-serif;
    background: #f4f7f4;
    margin: 0;
    padding: 40px;
}

.container {
    max-width: 850px;
    margin: auto;
    background: white;
    padding: 35px;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}

h1 {
    color: #246b3c;
}

.upload-box {
    padding: 25px;
    border: 2px dashed #8aaa8f;
    border-radius: 12px;
    text-align: center;
    margin: 25px 0;
}

button {
    background: #246b3c;
    color: white;
    border: none;
    padding: 12px 25px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 16px;
}

button:hover {
    background: #18532d;
}

.result {
    margin-top: 30px;
    padding: 25px;
    background: #eef7ee;
    border-radius: 12px;
}

.top-result {
    font-size: 24px;
    font-weight: bold;
    color: #246b3c;
}

.prediction {
    padding: 12px;
    margin: 8px 0;
    background: white;
    border-radius: 8px;
}

.warning {
    margin-top: 20px;
    padding: 12px;
    background: #fff4d6;
    border-radius: 8px;
    font-size: 14px;
}

</style>

</head>


<body>

<div class="container">

<h1>
🌱 Bottle Gourd Disease Detection
</h1>

<p>
Upload a Bottle Gourd leaf image for AI-based disease classification.
</p>


<div class="upload-box">

<form
method="POST"
action="/predict"
enctype="multipart/form-data"
>

<input
type="file"
name="image"
accept=".jpg,.jpeg,.png,.webp"
required
>

<br><br>

<button type="submit">
Analyze Leaf
</button>

</form>

</div>


{% if predictions %}

<div class="result">

<div class="top-result">

Prediction:
{{ predictions[0]["info"]["name"] }}

</div>

<p>
Hindi:
{{ predictions[0]["info"]["hindi"] }}
</p>

<p>
Model confidence:
{{ "%.2f"|format(predictions[0]["confidence"] * 100) }}%
</p>


<h3>
Top 3 Predictions
</h3>


{% for p in predictions %}

<div class="prediction">

<b>
{{ loop.index }}.
{{ p["info"]["name"] }}
</b>

—
{{ "%.2f"|format(p["confidence"] * 100) }}%

</div>

{% endfor %}


<h3>
Farmer Guidance
</h3>

<p>
{{ predictions[0]["info"]["advice"] }}
</p>


<div class="warning">

⚠️ This is an AI-based image classification result.
The confidence value is a model score and should not
be treated as a guaranteed medical/agricultural diagnosis.
For important crop-management decisions, verify the
symptoms with an agricultural expert.

</div>

</div>

{% endif %}

</div>

</body>

</html>
"""


# ============================================================
# ROUTES
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return render_template_string(
        HTML,
        predictions=None
    )


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return render_template_string(
            HTML,
            predictions=None
        )

    file = request.files["image"]

    if file.filename == "":
        return render_template_string(
            HTML,
            predictions=None
        )

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        return (
            "Unsupported image format",
            400
        )

    filename = secure_filename(
        file.filename
    )

    save_path = (
        UPLOAD_DIR / filename
    )

    file.save(save_path)

    predictions = predict_image(
        save_path
    )

    return render_template_string(
        HTML,
        predictions=predictions
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("SERVER STARTING")
    print("=" * 70)

    print(
        "Open your browser at:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print("=" * 70)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )