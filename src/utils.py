from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

TRAIN_JSON_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def seed_everything(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(root, rel):
    p = Path(rel)
    return p if p.is_absolute() else Path(root) / p


def observation_key(file_name):
    return Path(file_name).stem


def image_files(folder):
    """Image files of a folder, extension matched case-insensitively, sorted by name."""
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def find_data_root(configured=None, search_roots=("/kaggle/input", "/kaggle/working", ".", "data")):
    """Locate ``MAGFiLO_1.0_Kaggle_2026`` (the folder that contains train/ and test/).

    Order: ``$FILANET_DATA_ROOT``, the configured path, then a search for the
    official training JSON under the usual Kaggle/local locations (Kaggle has
    mounted competition data both at /kaggle/input/<slug>/ and at
    /kaggle/input/competitions/<slug>/).
    """
    candidates = []
    if os.environ.get("FILANET_DATA_ROOT"):
        candidates.append(Path(os.environ["FILANET_DATA_ROOT"]))
    if configured:
        candidates.append(Path(configured))
    for c in candidates:
        if (c / "train" / TRAIN_JSON_NAME).exists():
            return c
    for base in search_roots:
        base = Path(base)
        if not base.exists():
            continue
        for depth_glob in (f"train/{TRAIN_JSON_NAME}", f"*/train/{TRAIN_JSON_NAME}",
                           f"*/*/train/{TRAIN_JSON_NAME}", f"*/*/*/train/{TRAIN_JSON_NAME}"):
            hits = sorted(base.glob(depth_glob))
            if hits:
                return hits[0].parent.parent
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"could not find {TRAIN_JSON_NAME}; set data.root in the config or $FILANET_DATA_ROOT"
    )


def data_paths(cfg):
    """Resolved (root, train_json, train_images, test_images) for a config."""
    root = find_data_root(cfg["data"].get("root"))
    return (
        root,
        resolve_path(root, cfg["data"]["train_json"]),
        resolve_path(root, cfg["data"]["train_images"]),
        resolve_path(root, cfg["data"]["test_images"]),
    )
