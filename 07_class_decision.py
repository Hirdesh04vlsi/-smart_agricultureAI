import csv
from pathlib import Path
from collections import defaultdict

INPUT_FILE = Path("original_bottlegourd_inventory.csv")
OUTPUT_FILE = Path("class_decision_table.csv")

# ------------------------------------------------------------
# READ INVENTORY
# ------------------------------------------------------------

classes = defaultdict(int)

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:

        key = (
            row["dataset"],
            row["original_class"]
        )

        classes[key] += 1


# ------------------------------------------------------------
# CREATE TABLE
# ------------------------------------------------------------

rows = []

for (dataset, original_class), count in sorted(classes.items()):

    rows.append({
        "dataset": dataset,
        "original_class": original_class,
        "image_count": count,
        "use_for_disease_model": "",
        "standardized_class": "",
        "reason": ""
    })


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
        "image_count",
        "use_for_disease_model",
        "standardized_class",
        "reason"
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(rows)


print()
print("======================================")
print("CLASS DECISION TABLE CREATED")
print("======================================")

print(f"Total class entries: {len(rows)}")

print()
print("Saved to:")
print(OUTPUT_FILE.resolve())

print()
print("NO FILES WERE MODIFIED.")