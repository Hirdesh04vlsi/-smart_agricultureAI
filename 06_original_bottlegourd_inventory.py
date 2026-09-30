import csv
from pathlib import Path

INPUT_FILE = Path("dataset_inventory.csv")
OUTPUT_FILE = Path("original_bottlegourd_inventory.csv")

records = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:

        # Keep ONLY original/non-augmented images
        if row["augmented"].upper() == "NO":

            records.append(row)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "dataset",
        "original_class",
        "filename",
        "source_path",
        "extension",
        "augmented"
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(records)


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

dataset_counts = {}

class_counts = {}

for row in records:

    dataset = row["dataset"]
    cls = row["original_class"]

    dataset_counts[dataset] = \
        dataset_counts.get(dataset, 0) + 1

    key = (dataset, cls)

    class_counts[key] = \
        class_counts.get(key, 0) + 1


print()
print("======================================")
print("ORIGINAL BOTTLE GOURD INVENTORY")
print("======================================")

print(f"Total original images: {len(records)}")

print()
print("BY DATASET:")

for dataset, count in sorted(dataset_counts.items()):

    print(f"  {dataset}: {count}")

print()
print("BY DATASET + ORIGINAL CLASS:")

for (dataset, cls), count in sorted(class_counts.items()):

    print(
        f"  {dataset} | "
        f"{cls}: {count}"
    )

print()
print("Saved to:")
print(OUTPUT_FILE.resolve())

print()
print("NO FILES WERE MODIFIED.")