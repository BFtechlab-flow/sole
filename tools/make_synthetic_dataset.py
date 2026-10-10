"""Generate a tiny SYNTHETIC dataset with the official MAGFiLO/Kaggle layout.

It exists only to smoke-test the code paths (CI, the notebook's SMOKE mode):
images are artificial dark curves on a limb-darkened disk, NOT solar data, and
nothing computed on them says anything about competition performance.

Layout produced (same names as the competition)::

    <out>/train/train_images/*.jpg
    <out>/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
    <out>/test/test_images/*.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

TRAIN_JSON = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"


def _filament(rng, size):
    """Random smooth polyline + half-width -> (polygon xy list, spine points)."""
    h = w = size
    r0 = 0.30 * size
    ang0 = rng.uniform(0, 2 * np.pi)
    cx, cy = w / 2 + r0 * rng.uniform(0, 1) * np.cos(ang0), h / 2 + r0 * rng.uniform(0, 1) * np.sin(ang0)
    n = 10
    length = rng.uniform(0.12, 0.30) * size
    ang = rng.uniform(0, np.pi)
    t = np.linspace(-0.5, 0.5, n)
    bend = rng.normal(0, 0.06 * size)
    spine = np.stack([cx + t * length * np.cos(ang) - bend * (t ** 2) * np.sin(ang),
                      cy + t * length * np.sin(ang) + bend * (t ** 2) * np.cos(ang)], 1)
    half = rng.uniform(0.006, 0.016) * size
    d = np.gradient(spine, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    normal = np.stack([-d[:, 1], d[:, 0]], 1)
    left, right = spine + half * normal, spine - half * normal
    poly = np.concatenate([left, right[::-1], left[:1]], 0)
    poly = np.clip(poly, 0, size - 1)
    return poly.ravel().round(2).tolist(), spine


def _image(rng, size, polys):
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    r = np.hypot(xx - size / 2, yy - size / 2) / (0.45 * size)
    disk = np.clip(1 - r ** 2, 0, 1) ** 0.25
    img = 150 * disk + rng.normal(0, 6, (size, size))
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.5)
    dark = np.zeros((size, size), np.uint8)
    for p in polys:
        pts = np.asarray(p, np.float32).reshape(-1, 2).round().astype(np.int32)
        cv2.fillPoly(dark, [pts], 1)
    dark = cv2.GaussianBlur(dark.astype(np.float32), (0, 0), 1.0)
    img = img * (1 - 0.45 * dark)
    img[r > 1] = 0
    return np.clip(img, 0, 255).astype(np.uint8)


def make_dataset(out, n_train=8, n_test=3, size=256, seed=0):
    rng = np.random.default_rng(seed)
    out = Path(out)
    (out / "train" / "train_images").mkdir(parents=True, exist_ok=True)
    (out / "test" / "test_images").mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    aid = 0
    for i in range(n_train + n_test):
        stem = f"20{11 + i % 12:02d}{1 + i % 12:02d}{1 + i % 27:02d}1727{i % 60:02d}{'BLMCTU'[i % 6]}h"
        polys = [_filament(rng, size)[0] for _ in range(int(rng.integers(2, 6)))]
        img = _image(rng, size, polys)
        if i >= n_train:
            cv2.imwrite(str(out / "test" / "test_images" / f"{stem}.jpg"), img)
            continue
        cv2.imwrite(str(out / "train" / "train_images" / f"{stem}.jpg"), img)
        for a in range(1 + int(rng.random() < 0.5)):  # some observations have 2 annotators
            entry = f"0101{a + 1:02d}-{stem}"
            images.append({"id": entry, "width": size, "height": size, "file_name": f"{stem}.jpg",
                           "license": 1, "date_captured": None})
            for p in polys:
                if a and rng.random() < 0.25:
                    continue  # annotator disagreement
                aid += 1
                xy = np.asarray(p).reshape(-1, 2)
                x0, y0 = xy.min(0)
                x1, y1 = xy.max(0)
                annotations.append({"id": f"synthetic-{aid}", "image_id": entry, "category_id": int(rng.integers(1, 5)),
                                    "segmentation": [p], "area": 0.0, "spine": [],
                                    "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)], "iscrowd": 0})
    coco = {"info": {"description": "SYNTHETIC smoke-test data, not MAGFiLO"}, "images": images,
            "annotations": annotations, "licenses": [], "categories": [
                {"id": 1, "name": "Left"}, {"id": 2, "name": "Right"},
                {"id": 3, "name": "Unidentifiable"}, {"id": 4, "name": "Ambiguous"}]}
    (out / "train" / TRAIN_JSON).write_text(json.dumps(coco), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/synthetic_magfilo")
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--n-test", type=int, default=3)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(make_dataset(a.out, a.n_train, a.n_test, a.size, a.seed))


if __name__ == "__main__":
    main()
