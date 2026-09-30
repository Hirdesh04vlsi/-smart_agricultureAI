from pathlib import Path

# ============================================================
# MASTER DATASET LOCATION
# ============================================================

MASTER_ROOT = Path("MASTER_DATASET")

# ============================================================
# MODEL V1 CLASSES
# ============================================================

CLASSES = [
    "Healthy",
    "Anthracnose",
    "Downy_Mildew",
    "Alternaria_Leaf_Blight",
    "Early_Alternaria_Leaf_Blight",
    "Fungal_Damage_Leaf",
    "Mosaic_Virus",
    "Dry_Leaf",
    "Nutrition_Deficiency",
    "Pest_Infestation"
]

# ============================================================
# CREATE FOLDERS
# ============================================================

MASTER_ROOT.mkdir(exist_ok=True)

for class_name in CLASSES:

    class_folder = MASTER_ROOT / class_name

    class_folder.mkdir(
        parents=True,
        exist_ok=True
    )

# ============================================================
# VERIFY
# ============================================================

print()
print("======================================")
print("MASTER DATASET FOLDERS CREATED")
print("======================================")

print()

for class_name in CLASSES:

    folder = MASTER_ROOT / class_name

    if folder.exists():
        print(f"[OK] {class_name}")
    else:
        print(f"[ERROR] {class_name}")

print()
print(f"Location:")
print(MASTER_ROOT.resolve())

print()
print(f"Total class folders: {len(CLASSES)}")

print()
print("NO IMAGES WERE COPIED.")
print("NO ORIGINAL FILES WERE MODIFIED.")