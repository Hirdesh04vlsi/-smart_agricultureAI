"""Evaluate a saved Bottle Gourd EfficientNet classifier on SPLIT_DATASET/test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from efficientnet_utils import create_model, image_transform


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "TRAINING_OUTPUT" / "best_efficientnet_b0.pt",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=PROJECT_ROOT / "SPLIT_DATASET" / "test",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "TRAINING_OUTPUT" / "test_metrics.json",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    classes = checkpoint["class_names"]
    dataset = datasets.ImageFolder(args.test_dir, transform=image_transform(False))
    if dataset.classes != classes:
        raise ValueError("Test classes do not match the checkpoint class names.")

    model = create_model(len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    confusion = [[0 for _ in classes] for _ in classes]
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            for actual, predicted in zip(labels, predictions):
                confusion[actual.item()][predicted.item()] += 1

    correct = sum(confusion[index][index] for index in range(len(classes)))
    total = sum(sum(row) for row in confusion)
    metrics = {
        "images": total,
        "accuracy": correct / total if total else 0.0,
        "classes": classes,
        "confusion_matrix": confusion,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Test accuracy: {metrics['accuracy']:.2%} ({correct}/{total})")
    print(f"Metrics saved to: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

