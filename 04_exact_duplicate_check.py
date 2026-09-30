import os
import csv
import hashlib
from pathlib import Path

RAW_ROOT = Path(
    r"D:\AI_Agriculture_Dataset\RAW_DATASETS\BOTTLE_GOURD"
)

OUTPUT_FILE = Path("exact_duplicates.csv")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}


def sha256_file(path):

    sha = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)

    return sha.hexdigest()


hash_map = {}

total = 0

print("Scanning images for exact duplicates...")
print()

for root, dirs, files in os.walk(RAW_ROOT):

    for file in files:

        path = Path(root) / file

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        total += 1

        try:
            file_hash = sha256_file(path)

            if file_hash not in hash_map:
                hash_map[file_hash] = []

            hash_map[file_hash].append(path)

        except Exception as e:

            print(f"ERROR: {path}")
            print(e)


duplicate_groups = []

for file_hash, paths in hash_map.items():

    if len(paths) > 1:

        duplicate_groups.append({
            "sha256": file_hash,
            "count": len(paths),
            "files": paths
        })


with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "sha256",
        "duplicate_count",
        "file"
    ])

    for group in duplicate_groups:

        for path in group["files"]:

            writer.writerow([
                group["sha256"],
                group["count"],
                str(path.resolve())
            ])


print()
print("======================================")
print("EXACT DUPLICATE CHECK COMPLETE")
print("======================================")

print(f"Total images scanned: {total}")
print(f"Duplicate groups found: {len(duplicate_groups)}")

duplicate_images = sum(
    group["count"]
    for group in duplicate_groups
)

print(f"Images involved in duplicate groups: {duplicate_images}")

print()
print(f"Report saved to:")
print(OUTPUT_FILE.resolve())

print()
print("NO FILES WERE DELETED OR MODIFIED.")