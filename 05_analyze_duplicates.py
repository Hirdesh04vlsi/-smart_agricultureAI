import csv
from pathlib import Path
from collections import defaultdict

INPUT_FILE = Path("exact_duplicates.csv")
OUTPUT_FILE = Path("duplicate_group_analysis.csv")

groups = defaultdict(list)

# ------------------------------------------------------------
# READ DUPLICATE REPORT
# ------------------------------------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:

        sha = row["sha256"]
        file_path = row["file"]

        groups[sha].append(file_path)


# ------------------------------------------------------------
# ANALYZE EACH GROUP
# ------------------------------------------------------------

results = []

for group_id, (sha, files) in enumerate(groups.items(), start=1):

    datasets = set()

    for file_path in files:

        parts = Path(file_path).parts

        dataset = "UNKNOWN"

        for part in parts:

            if part.startswith("BG01_"):
                dataset = "BG01_Mendeley"

            elif part.startswith("BG02_"):
                dataset = "BG02_AgriVision"

            elif part.startswith("BG03_"):
                dataset = "BG03_Freshness"

            elif part.startswith("BG04_"):
                dataset = "BG04_Anthracnose"

            elif part.startswith("BG05_"):
                dataset = "BG05_CAIR"

        datasets.add(dataset)

    if len(datasets) == 1:
        duplicate_type = "WITHIN_DATASET"
    else:
        duplicate_type = "CROSS_DATASET"

    results.append({
        "group_id": group_id,
        "sha256": sha,
        "number_of_copies": len(files),
        "duplicate_type": duplicate_type,
        "datasets": " | ".join(sorted(datasets)),
        "files": " || ".join(files)
    })


# ------------------------------------------------------------
# SAVE ANALYSIS
# ------------------------------------------------------------

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "group_id",
            "sha256",
            "number_of_copies",
            "duplicate_type",
            "datasets",
            "files"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

within = 0
cross = 0

within_images = 0
cross_images = 0

for r in results:

    copies = int(r["number_of_copies"])

    if r["duplicate_type"] == "WITHIN_DATASET":

        within += 1
        within_images += copies

    else:

        cross += 1
        cross_images += copies


print()
print("======================================")
print("DUPLICATE GROUP ANALYSIS")
print("======================================")

print(f"Total duplicate groups: {len(results)}")

print()
print("Within same dataset:")
print(f"  Groups : {within}")
print(f"  Images : {within_images}")

print()
print("Across different datasets:")
print(f"  Groups : {cross}")
print(f"  Images : {cross_images}")

print()
print("Analysis saved to:")
print(OUTPUT_FILE.resolve())

print()
print("NO FILES WERE DELETED OR MODIFIED.")