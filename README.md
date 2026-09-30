# 🌱 AI-Based Bottle Gourd Leaf Disease Detection

An AI-powered plant disease detection system for **Bottle Gourd (Lagenaria siceraria)** using deep learning, classical machine learning, image analysis, and a Flask-based web application.

The project develops and evaluates an **EfficientNet-B0 image classification model** for identifying Bottle Gourd leaf conditions and compares its performance with a **Random Forest + GLCM texture/color feature baseline**.

---

## 📌 Project Overview

Plant diseases can significantly affect crop productivity, particularly when disease symptoms are detected at a late stage.

This project develops an image-based disease detection pipeline for Bottle Gourd leaves. The complete workflow includes:

1. Dataset collection and inventory
2. Dataset cleaning and duplicate detection
3. Disease-class standardization
4. Master dataset construction
5. Stratified train/validation/test splitting
6. Image resizing and preprocessing
7. EfficientNet-B0 model training
8. Classical Random Forest + GLCM baseline
9. Held-out test evaluation
10. Error and confusion analysis
11. Cross-source validation
12. Publication-oriented visualization
13. Flask-based web inference application
14. Disease-specific farmer recommendations

The current implementation focuses on **Bottle Gourd** as the first crop in a larger planned multi-crop agricultural AI system.

---

# 🎯 Objectives

The main objectives of this project are:

- Develop an image-based Bottle Gourd disease classification system.
- Combine images from multiple publicly available datasets.
- Standardize disease labels across different data sources.
- Remove exact duplicate images before model development.
- Evaluate a modern deep learning model against a classical ML baseline.
- Analyze model errors rather than relying only on overall accuracy.
- Evaluate performance under a cross-source/domain-shift scenario.
- Deploy the trained model through a web-based interface.
- Provide bilingual disease information and prevention recommendations.

---

# 🌿 Current Crop

### Bottle Gourd

**Common name:** Bottle Gourd  
**Hindi:** लौकी  
**Scientific name:** *Lagenaria siceraria*

The current model recognizes the following 10 classes:

| No. | Disease / Condition |
|---:|---|
| 1 | Alternaria Leaf Blight |
| 2 | Anthracnose |
| 3 | Downy Mildew |
| 4 | Dry Leaf |
| 5 | Early Alternaria Leaf Blight |
| 6 | Fungal Damage Leaf |
| 7 | Healthy |
| 8 | Mosaic Virus |
| 9 | Nutrition Deficiency |
| 10 | Pest Infestation |

---

# 🗂️ Dataset Development

Multiple Bottle Gourd image datasets were investigated and combined.

The final master dataset was constructed after:

- Dataset inventory
- Class mapping
- Duplicate analysis
- Exclusion of unsuitable/general classes
- Exact duplicate removal
- Standardized naming
- Metadata generation

## Final Dataset

The master dataset contains:

**12,104 images**

distributed across 10 standardized classes.

### Final class distribution

| Class | Images |
|---|---:|
| Healthy | 2,886 |
| Anthracnose | 2,229 |
| Downy Mildew | 1,976 |
| Mosaic Virus | 1,216 |
| Alternaria Leaf Blight | 1,098 |
| Early Alternaria Leaf Blight | 1,011 |
| Fungal Damage Leaf | 1,009 |
| Dry Leaf | 363 |
| Pest Infestation | 216 |
| Nutrition Deficiency | 100 |
| **Total** | **12,104** |

---

# 🧹 Dataset Cleaning

An exact duplicate analysis was performed across the collected datasets.

The analysis identified duplicate image groups and duplicate copies originating primarily from augmented datasets.

The final master dataset contains:

- **12,104 usable images**
- **357 duplicate copies skipped during master construction**
- Raw source datasets were kept unchanged.

The dataset metadata is maintained in:

```text
master_dataset_metadata.csv
