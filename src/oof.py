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
)


def cache_path(cache_dir, fold: int, file_name: str) -> Path:
    return Path(cache_dir) / f"fold{int(fold)}" / f"{Path(file_name).stem}.npz"


def save_prediction(path, prediction: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for key in PREDICTION_KEYS:
        if key in prediction:
            arrays[key] = np.asarray(prediction[key], dtype=np.float16)
    np.savez_compressed(path, **arrays)


def load_prediction(path) -> dict:
    with np.load(path) as data:
        return {key: data[key].astype(np.float32) for key in data.files}


def available_prediction(path) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    try:
        with np.load(path) as data:
            return all(k in data.files for k in ("region", "centerline", "boundary", "distance"))
    except Exception:
        return False
