from pathlib import Path
import random
import shutil

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path.cwd()

SOURCE = ROOT / "MASTER_DATASET"
OUTPUT = ROOT / "SPLIT_DATASET"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

# ============================================================
# CHECK RATIOS
# ============================================================

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6

random.seed(SEED)

# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

for split in ["train", "val", "test"]:
    (OUTPUT / split).mkdir(parents=True, exist_ok=True)

# ============================================================
# GET CLASSES
# ============================================================

classes = sorted([
    folder.name
    for folder in SOURCE.iterdir()
    if folder.is_dir()
])

print("=" * 60)
print("BOTTLE GOURD DATASET SPLIT")
print("=" * 60)

print(f"\nSource: {SOURCE}")
print(f"Output: {OUTPUT}")
print(f"\nClasses found: {len(classes)}")

for cls in classes:
    print(" -", cls)

# ============================================================
# SPLIT EACH CLASS
# ============================================================

total_train = 0
total_val = 0
total_test = 0

for cls in classes:

    source_class = SOURCE / cls

    images = [
        f for f in source_class.iterdir()
        if f.is_file()
        and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ]

    random.shuffle(images)

    n = len(images)

    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    # Create class directories
    train_class = OUTPUT / "train" / cls
    val_class = OUTPUT / "val" / cls
    test_class = OUTPUT / "test" / cls

    train_class.mkdir(parents=True, exist_ok=True)
    val_class.mkdir(parents=True, exist_ok=True)
    test_class.mkdir(parents=True, exist_ok=True)

    # Copy files
    for img in train_images:
        shutil.copy2(img, train_class / img.name)

    for img in val_images:
        shutil.copy2(img, val_class / img.name)

    for img in test_images:
        shutil.copy2(img, test_class / img.name)

    total_train += len(train_images)
    total_val += len(val_images)
    total_test += len(test_images)

    print(
        f"{cls:35s} "
        f"Total={n:5d} | "
        f"Train={len(train_images):5d} | "
        f"Val={len(val_images):5d} | "
        f"Test={len(test_images):5d}"
    )

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SPLIT COMPLETE")
print("=" * 60)

print(f"Train images : {total_train}")
print(f"Validation   : {total_val}")
print(f"Test images  : {total_test}")
print(f"Total        : {total_train + total_val + total_test}")

print("\nSplit ratio:")
print(f"Train: {total_train / (total_train + total_val + total_test):.2%}")
print(f"Val  : {total_val / (total_train + total_val + total_test):.2%}")
print(f"Test : {total_test / (total_train + total_val + total_test):.2%}")

print("\nDataset location:")
print(OUTPUT)

print("\nRaw MASTER_DATASET was NOT modified.")