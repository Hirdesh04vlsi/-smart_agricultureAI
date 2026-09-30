import os
import csv
import hashlib
from pathlib import Path
from PIL import Image
import imagehash

# ============================================================
# CONFIGURATION
# ============================================================

RAW_ROOT = Path(
    r"D:\AI_Agriculture_Dataset\RAW_DATASETS\BOTTLE_GOURD"
)

EXACT_REPORT = Path("exact_duplicates.csv")
NEAR_REPORT = Path("near_duplicates.csv")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}

# Maximum perceptual-hash distance considered near-duplicate
PHASH_THRESHOLD = 5


# ============================================================
# SHA-256
# ============================================================

def sha256_file(path):

    sha = hashlib.sha256()

    with open(path, "rb") as f:

        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)

    return sha.hexdigest()


# ============================================================
# COLLECT IMAGES
# ============================================================

images = []

print("Scanning images...")

for root, dirs, files in os.walk(RAW_ROOT):

    for file in files:

        path = Path(root) / file

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        images.append(path)

print()
print("======================================")
print("IMAGE COLLECTION COMPLETE")
print("======================================")
print(f"Total images found: {len(images)}")


# ============================================================
# EXACT DUPLICATES
# ============================================================

print()
print("Checking exact duplicates...")

hash_map = {}

for i, path in enumerate(images, start=1):

    try:

        file_hash = sha256_file(path)

        if file_hash not in hash_map:
            hash_map[file_hash] = []

        hash_map[file_hash].append(path)

    except Exception as e:

        print(f"ERROR: {path}")
        print(e)

exact_duplicates = []

for file_hash, paths in hash_map.items():

    if len(paths) > 1:

        for path in paths:

            exact_duplicates.append({
                "sha256": file_hash,
                "file": str(path.resolve())
            })


with open(
    EXACT_REPORT,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "sha256",
            "file"
        ]
    )

    writer.writeheader()
    writer.writerows(exact_duplicates)


# ============================================================
# PERCEPTUAL HASH
# ============================================================

print()
print("Calculating perceptual hashes...")

phashes = []

for i, path in enumerate(images, start=1):

    try:

        with Image.open(path) as img:

            img = img.convert("RGB")

            phash = imagehash.phash(img)

            phashes.append({
                "path": path,
                "hash": phash
            })

    except Exception as e:

        print(f"ERROR reading: {path}")
        print(e)


# ============================================================
# NEAR DUPLICATES
# ============================================================

print()
print("Checking near-duplicates...")

near_duplicates = []

for i in range(len(phashes)):

    hash1 = phashes[i]["hash"]
    path1 = phashes[i]["path"]

    for j in range(i + 1, len(phashes)):

        hash2 = phashes[j]["hash"]
        path2 = phashes[j]["path"]

        distance = hash1 - hash2

        if distance <= PHASH_THRESHOLD:

            near_duplicates.append({
                "distance": distance,
                "file_1": str(path1.resolve()),
                "file_2": str(path2.resolve())
            })


with open(
    NEAR_REPORT,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "distance",
            "file_1",
            "file_2"
        ]
    )

    writer.writeheader()
    writer.writerows(near_duplicates)


# ============================================================
# SUMMARY
# ============================================================

print()
print("======================================")
print("DUPLICATE CHECK COMPLETE")
print("======================================")

print(
    f"Exact duplicate records: "
    f"{len(exact_duplicates)}"
)

print(
    f"Near-duplicate pairs: "
    f"{len(near_duplicates)}"
)

print()
print(f"Exact report: {EXACT_REPORT.resolve()}")
print(f"Near report:  {NEAR_REPORT.resolve()}")

print()
print("NO FILES WERE DELETED OR MODIFIED.")