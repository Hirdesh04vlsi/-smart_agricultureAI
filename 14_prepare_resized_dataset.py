from pathlib import Path
from PIL import Image
import shutil

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

SOURCE = ROOT / "SPLIT_DATASET"
OUTPUT = ROOT / "RESIZED_DATASET"

IMAGE_SIZE = (224, 224)

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PREPARING RESIZED DATASET")
    print("=" * 70)

    print("\nSource:")
    print(SOURCE)

    print("\nOutput:")
    print(OUTPUT)

    total = 0
    errors = 0

    splits = [
        "train",
        "val",
        "test"
    ]

    for split in splits:

        source_split = SOURCE / split
        output_split = OUTPUT / split

        print(
            f"\nProcessing: {split}"
        )

        for class_dir in sorted(
            source_split.iterdir()
        ):

            if not class_dir.is_dir():
                continue

            output_class = (
                output_split /
                class_dir.name
            )

            output_class.mkdir(
                parents=True,
                exist_ok=True
            )

            images = [
                p for p in class_dir.iterdir()
                if p.is_file()
                and p.suffix.lower()
                in VALID_EXTENSIONS
            ]

            print(
                f"  {class_dir.name}: "
                f"{len(images)} images"
            )

            for image_path in images:

                output_path = (
                    output_class /
                    image_path.name
                )

                try:

                    with Image.open(
                        image_path
                    ) as img:

                        img = img.convert(
                            "RGB"
                        )

                        img = img.resize(
                            IMAGE_SIZE,
                            Image.Resampling.LANCZOS
                        )

                        # Save as JPEG for faster loading
                        output_path = (
                            output_path.with_suffix(
                                ".jpg"
                            )
                        )

                        img.save(
                            output_path,
                            "JPEG",
                            quality=95
                        )

                    total += 1

                except Exception as e:

                    errors += 1

                    print(
                        "\nERROR:",
                        image_path
                    )

                    print(
                        e
                    )

    print("\n")
    print("=" * 70)
    print("RESIZED DATASET COMPLETE")
    print("=" * 70)

    print(
        "\nImages processed:",
        total
    )

    print(
        "Errors:",
        errors
    )

    print(
        "\nDataset saved at:"
    )

    print(
        OUTPUT
    )

    print(
        "\nOriginal SPLIT_DATASET was NOT modified."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()