"""Lightweight ground-truth helpers for evaluation paths."""
from __future__ import annotations

from .data import segmentation_to_mask


def build_instances(record):
    """Decode only instance masks, avoiding expensive auxiliary-target creation."""
    h, w = int(record["height"]), int(record["width"])
    instances = []
    for annotation in record["annotations"]:
        mask = segmentation_to_mask(annotation["segmentation"], h, w)
        if mask.any():
            instances.append(mask)
    return instances
