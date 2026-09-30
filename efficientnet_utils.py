"""Shared model and preprocessing helpers for the Bottle Gourd classifier."""

from __future__ import annotations

from typing import Any

from torch import nn
from torchvision import models, transforms


def image_transform(training: bool) -> transforms.Compose:
    steps: list[Any] = [transforms.Resize((224, 224))]
    if training:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transforms.Compose(steps)


def create_model(number_of_classes: int, pretrained: bool) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    try:
        model = models.efficientnet_b0(weights=weights)
    except Exception as error:
        if pretrained:
            raise RuntimeError(
                "Pretrained EfficientNet weights could not be loaded. If the "
                "download was interrupted, remove the specific corrupt cache "
                "file named in the error and retry, or run without --pretrained."
            ) from error
        raise

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, number_of_classes
    )
    return model

