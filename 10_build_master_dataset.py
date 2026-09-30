import csv
import shutil
import hashlib
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT = Path.cwd()

INPUT_FILE = ROOT / "mapped_inventory.csv"

MASTER_ROOT = ROOT / "MASTER_DATASET"

OUTPUT_METADATA = ROOT / "master_dataset_metadata.csv"


# ============================================================
# EXACT DUPLICATE TRACKING
# ============================================================

copied_hashes = set()


# ============================================================
# STATISTICS
# ============================================================

copied_count = 0
skipped_count = 0
duplicate_skipped = 0
error_count = 0


# ============================================================
# SHA-256
# ============================================================

def sha256_file(path):

    sha = hashlib.sha256()

    with open(path, "rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            sha.update(block)

    return sha.hexdigest()


# ============================================================
# READ MAPPED INVENTORY
# ============================================================

records = []

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        if row["use_for_disease_model"] == "USE":

            records.append(row)


print()
print("======================================")
print("BUILDING MASTER DATASET")
print("======================================")

print()
print(f"Images marked USE: {len(records)}")
print()


# ============================================================
# METADATA OUTPUT
# ============================================================

metadata_rows = []


# ============================================================
# COPY IMAGES
# ============================================================

for index, row in enumerate(records, start=1):

    source = Path(row["source_path"])

    dataset = row["dataset"]

    original_class = row["original_class"]

    standardized_class = row["standardized_class"]

    destination_folder = (
        MASTER_ROOT / standardized_class
    )

    destination_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # CHECK SOURCE
    # --------------------------------------------------------

    if not source.exists():

        print(
            f"ERROR: Source not found: {source}"
        )

        error_count += 1

        continue


    try:

        # ----------------------------------------------------
        # HASH
        # ----------------------------------------------------

        file_hash = sha256_file(source)


        # ----------------------------------------------------
        # DUPLICATE CHECK
        # ----------------------------------------------------

        if file_hash in copied_hashes:

            duplicate_skipped += 1

            continue


        copied_hashes.add(file_hash)


        # ----------------------------------------------------
        # SAFE NEW FILENAME
        # ----------------------------------------------------

        original_filename = source.name

        new_filename = (
            f"{dataset}__"
            f"{standardized_class}__"
            f"{original_filename}"
        )

        destination = (
            destination_folder /
            new_filename
        )


        # ----------------------------------------------------
        # COPY
        # ----------------------------------------------------

        shutil.copy2(
            source,
            destination
        )


        copied_count += 1


        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        metadata_rows.append({

            "image_id":
                f"BG_{copied_count:06d}",

            "dataset":
                dataset,

            "original_class":
                original_class,

            "standardized_class":
                standardized_class,

            "filename":
                new_filename,

            "original_filename":
                original_filename,

            "source_path":
                str(source.resolve()),

            "master_path":
                str(destination.resolve()),

            "sha256":
                file_hash,

            "augmented":
                row["augmented"]

        })


    except Exception as e:

        print(
            f"ERROR processing: {source}"
        )

        print(e)

        error_count += 1


# ============================================================
# SAVE METADATA
# ============================================================

fieldnames = [

    "image_id",
    "dataset",
    "original_class",
    "standardized_class",
    "filename",
    "original_filename",
    "source_path",
    "master_path",
    "sha256",
    "augmented"

]


with open(
    OUTPUT_METADATA,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(metadata_rows)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("======================================")
print("MASTER DATASET COMPLETE")
print("======================================")

print(
    f"Images marked USE: "
    f"{len(records)}"
)

print(
    f"Images copied: "
    f"{copied_count}"
)

print(
    f"Exact duplicate copies skipped: "
    f"{duplicate_skipped}"
)

print(
    f"Errors: "
    f"{error_count}"
)

print()
print(
    f"Metadata saved to:"
)

print(
    OUTPUT_METADATA.resolve()
)

print()
print(
    "Original RAW datasets were NOT modified."
)

print()
print(
    "Master dataset location:"
)

print(
    MASTER_ROOT.resolve()
)