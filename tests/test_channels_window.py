"""The faster data loading (windowed channels, per-worker caches) gives
bit-identical inputs and training samples to the original implementation."""
from __future__ import annotations

import random

import cv2
import numpy as np
import pytest
import torch

from src.data import TARGET_KEYS, FilamentDataset, annotation_masks, apply_geometric_transform, \
    build_input_channels, build_targets, input_channel_stats, load_coco_records


def legacy_build_input_channels(gray):
    """build_input_channels as used to train folds 0-1 (whole image, then cropped)."""
    x = gray.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99])
    raw = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
    u8 = (raw * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(u8).astype(np.float32) / 255
    b1 = cv2.GaussianBlur(raw, (0, 0), 1.2)
    b2 = cv2.GaussianBlur(raw, (0, 0), 5.0)
    ridge = b2 - b1
    ridge = (ridge - ridge.min()) / (np.ptp(ridge) + 1e-8)
    h, w = raw.shape
    yy, xx = np.mgrid[:h, :w]
    cx, cy = (w - 1) / 2, (h - 1) / 2
    radial = np.clip(np.sqrt(((xx - cx) / (cx + 1e-8)) ** 2 + ((yy - cy) / (cy + 1e-8)) ** 2), 0, 1)
    return np.stack([raw, clahe, ridge, radial], 0).astype(np.float32)


def _sun(size, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2
    disk = (yy - c) ** 2 + (xx - c) ** 2 < (0.44 * size) ** 2
    img = disk * (120 + 40 * np.sin(xx / 37.0) * np.cos(yy / 53.0)) + rng.normal(0, 8, (size, size))
    return img.clip(0, 255).astype(np.uint8)


@pytest.mark.parametrize("size,seed", [(2048, 0), (2048, 1), (700, 2)])
def test_window_channels_identical_to_whole_image(size, seed):
    img = _sun(size, seed)
    full = legacy_build_input_channels(img)
    assert np.array_equal(build_input_channels(img), full)
    stats = input_channel_stats(img)
    rng = np.random.default_rng(seed)
    s = min(1024, size)
    corners = [(0, 0), (size - s, size - s), (0, size - s), (size - s, 0)]
    corners += [tuple(int(v) for v in rng.integers(0, size - s + 1, 2)) for _ in range(6)]
    for y, x in corners:
        expected = full[:, y : y + s, x : x + s]
        assert np.array_equal(build_input_channels(img, window=(y, x, s, s), stats=stats), expected), (y, x)
        assert np.array_equal(build_input_channels(img, window=(y, x, s, s)), expected), (y, x)


def _legacy_getitem(ds, idx):
    """FilamentDataset.__getitem__ before the optimisation (reference)."""
    r = ds.records[idx]
    img = cv2.imread(str(ds.image_dir / r["file_name"]), cv2.IMREAD_GRAYSCALE)
    x = legacy_build_input_channels(img)
    masks = annotation_masks(r)
    h, w = int(r["height"]), int(r["width"])
    s = min(ds.patch_size, h, w)
    if ds.train:
        region = np.zeros((h, w), np.uint8)
        for m in masks:
            np.maximum(region, m, out=region)
        if region.any() and random.random() < 0.7:
            ys, xs = np.where(region > 0)
            j = random.randrange(len(xs))
            cy, cx = ys[j], xs[j]
            y = max(0, min(h - s, int(cy) - s // 2))
            z = max(0, min(w - s, int(cx) - s // 2))
        else:
            y = random.randint(0, max(h - s, 0))
            z = random.randint(0, max(w - s, 0))
    else:
        y = max((h - s) // 2, 0)
        z = max((w - s) // 2, 0)
    x = x[:, y : y + s, z : z + s]
    cropped = build_targets(r, width_scale_px=ds.width_scale_px, window=(y, z, s, s), masks=masks)
    cropped = {k: cropped[k] for k in TARGET_KEYS}
    if ds.train:
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        x, cropped = apply_geometric_transform(x, cropped, hflip=hflip, vflip=vflip)
    out = {"image": torch.from_numpy(np.ascontiguousarray(x)).float()}
    for k in TARGET_KEYS:
        v = cropped[k]
        if k == "orientation":
            out[k] = torch.from_numpy(v.copy()).float()
        elif k == "instance_id":
            out[k] = torch.from_numpy(v[None].copy()).long()
        else:
            out[k] = torch.from_numpy(v[None].copy()).float()
    return out


@pytest.mark.parametrize("train", [True, False])
def test_dataset_samples_identical_to_legacy(tmp_path, train):
    from tools.make_synthetic_dataset import make_dataset

    root = make_dataset(tmp_path / "d", n_train=3, n_test=1, size=512, seed=3)
    records = load_coco_records(root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    image_dir = root / "train" / "train_images"
    new = FilamentDataset(records, image_dir, patch_size=256, train=train)
    old = FilamentDataset(records, image_dir, patch_size=256, train=train)
    order = [0, 1, 2, 0, 1, 2, 2, 0]  # repeated indices exercise the caches
    random.seed(11)
    got = [new[i] for i in order]
    random.seed(11)
    ref = [_legacy_getitem(old, i) for i in order]
    for a, b in zip(got, ref):
        assert a.keys() == b.keys()
        for k in a:
            assert a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]), k
