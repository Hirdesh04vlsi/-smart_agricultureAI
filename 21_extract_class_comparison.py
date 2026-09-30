from pathlib import Path
import pandas as pd
import shutil

BASE = Path(r"D:\AI_Agriculture_Dataset\RAW_DATASETS\BOTTLE_GUARD_MASTER")

META = BASE / "master_dataset_metadata.csv"
OUT = BASE / "CROSS_SOURCE_RESULTS" / "CLASS_COMPARISON"

df = pd.read_csv(META)

# We want these comparisons
groups = {
    "BG01_Anthracnose": ("BG01_Mendeley", "Anthracnose"),
    "BG02_Anthracnose": ("BG02_AgriVision", "Anthracnose"),
    "BG03_Anthracnose": ("BG03_Freshness", "Anthracnose"),
    "BG05_Anthracnose": ("BG05_CAIRrlder", "Anthracnose"),

    "BG01_Alternaria": ("BG01_Mendeley", "Alternaria_Leaf_Blight"),
    "BG02_Alternaria": ("BG02_AgriVision", "Alternaria_Leaf_Blight"),
}

OUT.mkdir(parents=True, exist_ok=True)

for folder, (dataset, cls) in groups.items():

    subset = df[
        (df["dataset"] == dataset) &
        (df["standardized_class"] == cls)
    ].head(10)

    dest = OUT / folder
    dest.mkdir(parents=True, exist_ok=True)

    for _, row in subset.iterrows():
        src = Path(row["master_path"])

        if src.exists():
            shutil.copy2(
                src,
                dest / f"{row['image_id']}_{src.name}"
            )

    print(f"{folder}: {len(subset)} images")

print("\nDONE")
print(f"Output: {OUT}")