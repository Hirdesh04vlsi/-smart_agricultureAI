import csv
from pathlib import Path

INPUT_FILE = Path("original_bottlegourd_inventory.csv")
OUTPUT_FILE = Path("mapped_inventory.csv")


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_MAP = {

    # ----------------------------
    # Specific diseases
    # ----------------------------

    "Alternaria_Leaf_Blight":
        ("USE", "Alternaria_Leaf_Blight"),

    "Angular_Leaf_spot":
        ("EXCLUDE_LOW_SAMPLE", "Angular_Leaf_Spot"),

    "Anthracnose":
        ("USE", "Anthracnose"),

    "Downy_Mildew":
        ("USE", "Downy_Mildew"),

    "Downey mildew":
        ("USE", "Downy_Mildew"),

    "Downy Mildew":
        ("USE", "Downy_Mildew"),

    "Early_Alternaria_Leaf_Blight":
        ("USE", "Early_Alternaria_Leaf_Blight"),

    "Fungal_Damage_Leaf":
        ("USE", "Fungal_Damage_Leaf"),

    "Mosaic_Virus":
        ("USE", "Mosaic_Virus"),

    # ----------------------------
    # Healthy
    # ----------------------------

    "Healthy":
        ("USE", "Healthy"),

    "Fresh leaf":
        ("USE", "Healthy"),

    "Healthy Leaf":
        ("USE", "Healthy"),

    "Bottle  Gourd Healthy leaves":
        ("USE", "Healthy"),

    # ----------------------------
    # Other plant-health conditions
    # ----------------------------

    "Dry Leaf":
        ("USE", "Dry_Leaf"),

    "Nutrition Deficiency":
        ("USE", "Nutrition_Deficiency"),

    "Pest Infestation":
        ("USE", "Pest_Infestation"),

    # ----------------------------
    # BG04 generic disease severity
    # ----------------------------

    "Bottle  Gourd Diseased leaves":
        ("EXCLUDE", "Generic_Diseased"),

    "Bottle  Gourd Extreme Diseased leaves":
        ("EXCLUDE", "Generic_Extreme_Diseased"),

    # ----------------------------
    # Growth stages
    # ----------------------------

    "Flower":
        ("EXCLUDE", "Growth_Stage_Flower"),

    "Immature Gourd":
        ("EXCLUDE", "Growth_Stage_Immature"),

    "Mature Gourd":
        ("EXCLUDE", "Growth_Stage_Mature"),
}


# ============================================================
# READ ORIGINAL INVENTORY
# ============================================================

records = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:

        original_class = row["original_class"]

        if original_class not in CLASS_MAP:

            print(
                "WARNING: Unmapped class found:",
                original_class
            )

            action = "REVIEW"
            standardized_class = ""

        else:

            action, standardized_class = \
                CLASS_MAP[original_class]

        row["use_for_disease_model"] = action
        row["standardized_class"] = standardized_class

        records.append(row)


# ============================================================
# SAVE MAPPED INVENTORY
# ============================================================

fieldnames = [
    "dataset",
    "original_class",
    "standardized_class",
    "use_for_disease_model",
    "filename",
    "source_path",
    "extension",
    "augmented"
]

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(records)


# ============================================================
# SUMMARY
# ============================================================

use_counts = {}
exclude_counts = {}

for row in records:

    cls = row["standardized_class"]
    action = row["use_for_disease_model"]

    if action == "USE":

        use_counts[cls] = \
            use_counts.get(cls, 0) + 1

    elif action == "EXCLUDE":

        exclude_counts[cls] = \
            exclude_counts.get(cls, 0) + 1


print()
print("======================================")
print("MAPPED INVENTORY CREATED")
print("======================================")

print()
print("CLASSES FOR DISEASE/PLANT-HEALTH MODEL:")

for cls, count in sorted(use_counts.items()):

    print(f"  {cls}: {count}")


print()
print("EXCLUDED FROM FIRST MODEL:")

for cls, count in sorted(exclude_counts.items()):

    print(f"  {cls}: {count}")


print()
print("Total records:", len(records))

print()
print("Saved to:")
print(OUTPUT_FILE.resolve())

print()
print("NO ORIGINAL FILES WERE MODIFIED.")