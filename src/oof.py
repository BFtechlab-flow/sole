"""Compact on-disk cache of model outputs (OOF and test).

Full-resolution float16 maps of all heads take ~160 MB per image; the whole
OOF set would not fit on Kaggle's disk. Probabilities are stored as uint8
(step 1/255), axial orientation and the instance embedding as int8 (step
1/127); the embedding is zeroed where the region probability is below 0.05
(it is only read at skeleton endpoints, which are always inside the
foreground). The SAME quantisation is applied at test time
(:func:`quantize_roundtrip`), so post-processing tuned on the OOF cache sees
exactly the inputs it will see on the test set.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

PREDICTION_KEYS = (
    "region",
    "centerline",
    "boundary",
    "distance",
    "orientation",
    "width",
    "curvature",
    "endpoint",
    "junction",
    "uncertainty",
    "instance_embedding",
)
SIGNED_KEYS = ("orientation", "instance_embedding")
EMBEDDING_MIN_REGION = 0.05


def cache_path(cache_dir, fold: int, file_name: str) -> Path:
    return Path(cache_dir) / f"fold{int(fold)}" / f"{Path(file_name).stem}.npz"


def quantize(prediction: dict) -> dict:
    out = {}
    region = np.asarray(prediction["region"], dtype=np.float32) if "region" in prediction else None
    for key in PREDICTION_KEYS:
        if key not in prediction:
            continue
        x = np.asarray(prediction[key], dtype=np.float32)
        if key in SIGNED_KEYS:
            q = np.clip(np.rint(x * 127.0), -127, 127).astype(np.int8)
            if key == "instance_embedding" and region is not None:
                q[:, region < EMBEDDING_MIN_REGION] = 0
        else:
            q = np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)
        out[key] = q
    return out


def dequantize(arrays: dict) -> dict:
    out = {}
    for key, q in arrays.items():
        q = np.asarray(q)
        if q.dtype == np.uint8:
            out[key] = q.astype(np.float32) / 255.0
        elif q.dtype == np.int8:
            out[key] = q.astype(np.float32) / 127.0
        else:  # legacy float16 caches
            out[key] = q.astype(np.float32)
    return out


def quantize_roundtrip(prediction: dict) -> dict:
    """What the post-processing sees, whether the maps come from disk or not."""
    return dequantize(quantize(prediction))


def save_prediction(path, prediction: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **quantize(prediction))


RECONSTRUCTION_KEYS = ("region", "centerline", "boundary", "distance", "orientation", "width", "instance_embedding")


def load_prediction(path, keys=None) -> dict:
    """Load (and dequantise) a cached prediction; ``keys`` restricts what is read."""
    with np.load(path) as data:
        names = [k for k in data.files if keys is None or k in keys]
        return dequantize({key: data[key] for key in names})


def available_prediction(path) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    try:
        with np.load(path) as data:
            return all(k in data.files for k in ("region", "centerline", "boundary", "distance"))
    except Exception:
        return False
