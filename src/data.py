from collections import defaultdict
from pathlib import Path
import json
import random
import warnings

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from torch.utils.data import Dataset

from .utils import observation_key

TARGET_KEYS = (
    "region",
    "centerline",
    "boundary",
    "distance",
    "orientation",
    "orientation_valid",
)


def robust_normalize(img):
    x = img.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99])
    return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)


def build_input_channels(gray):
    raw = robust_normalize(gray)
    u8 = (raw * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(u8).astype(np.float32) / 255
    b1 = cv2.GaussianBlur(raw, (0, 0), 1.2)
    b2 = cv2.GaussianBlur(raw, (0, 0), 5.0)
    ridge = b2 - b1
    ridge = (ridge - ridge.min()) / (np.ptp(ridge) + 1e-8)
    h, w = raw.shape
    yy, xx = np.mgrid[:h, :w]
    cx, cy = (w - 1) / 2, (h - 1) / 2
    radial = np.clip(
        np.sqrt(((xx - cx) / (cx + 1e-8)) ** 2 + ((yy - cy) / (cy + 1e-8)) ** 2),
        0,
        1,
    )
    return np.stack([raw, clahe, ridge, radial], 0).astype(np.float32)


def segmentation_to_mask(seg, h, w):
    if isinstance(seg, list):
        r = mask_utils.merge(mask_utils.frPyObjects(seg, h, w))
    else:
        r = seg
    # pycocotools currently emits a NumPy 2.x copy-keyword deprecation from
    # inside its decoder. Keep the suppression tightly scoped to that call.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        m = mask_utils.decode(r)
    if m.ndim == 3:
        m = np.any(m, 2)
    return m.astype(np.uint8)


def load_coco_records(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    by = defaultdict(list)
    for a in data["annotations"]:
        by[a["image_id"]].append(a)
    return [
        {
            "image_id": i["id"],
            "file_name": i["file_name"],
            "height": int(i["height"]),
            "width": int(i["width"]),
            "annotations": by[i["id"]],
            "group": observation_key(i["file_name"]),
        }
        for i in data["images"]
    ]


def orientation_from_skeleton(skeleton, radius=3):
    """Return axial tangent field [cos(2θ), sin(2θ)] on valid skeleton pixels."""
    sk = np.asarray(skeleton, dtype=bool)
    h, w = sk.shape
    ox = np.zeros((h, w), np.float32)
    oy = np.zeros((h, w), np.float32)
    valid = np.zeros((h, w), np.float32)

    for y, x in np.argwhere(sk):
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        yy, xx = np.nonzero(sk[y0:y1, x0:x1])
        if len(xx) < 2:
            continue
        pts = np.column_stack((xx + x0, yy + y0)).astype(np.float32)
        pts -= pts.mean(axis=0, keepdims=True)
        cov = pts.T @ pts
        vals, vecs = np.linalg.eigh(cov)
        vx, vy = vecs[:, int(np.argmax(vals))]
        theta = np.arctan2(vy, vx)
        ox[y, x] = np.cos(2.0 * theta)
        oy[y, x] = np.sin(2.0 * theta)
        valid[y, x] = 1.0
    return np.stack([ox, oy], axis=0), valid


def build_targets(r):
    h, w = r["height"], r["width"]
    region = np.zeros((h, w), np.uint8)
    center = region.copy()
    boundary = region.copy()
    distance = np.zeros((h, w), np.float32)
    orientation_sum = np.zeros((2, h, w), np.float32)
    orientation_count = np.zeros((h, w), np.float32)
    instances = []
    kernel = np.ones((3, 3), np.uint8)

    for a in r["annotations"]:
        # Only segmentation GT is consumed. Auxiliary targets are derived.
        m = segmentation_to_mask(a["segmentation"], h, w)
        if not m.any():
            continue
        instances.append(m)
        region = np.maximum(region, m)

        skel = skeletonize(m > 0)
        center = np.maximum(
            center,
            cv2.dilate(skel.astype(np.uint8), kernel, iterations=1),
        )
        boundary = np.maximum(
            boundary,
            cv2.morphologyEx(m, cv2.MORPH_GRADIENT, kernel),
        )

        d = distance_transform_edt(m)
        d = d / (d.max() + 1e-8)
        distance = np.maximum(distance, d)

        orientation, valid = orientation_from_skeleton(skel)
        orientation_sum += orientation * valid[None]
        orientation_count += valid

    orientation_valid = (orientation_count > 0).astype(np.float32)
    orientation = np.zeros_like(orientation_sum)
    nz = orientation_count > 0
    if np.any(nz):
        orientation[:, nz] = orientation_sum[:, nz] / orientation_count[nz]
        norm = np.linalg.norm(orientation, axis=0)
        good = norm > 1e-6
        orientation[:, good] /= norm[good]

    return {
        "region": region.astype(np.float32),
        "centerline": center.astype(np.float32),
        "boundary": boundary.astype(np.float32),
        "distance": distance.astype(np.float32),
        "orientation": orientation.astype(np.float32),
        "orientation_valid": orientation_valid,
        "instances": instances,
    }


def apply_geometric_transform(image, targets, hflip=False, vflip=False):
    """Apply the same spatial transform to image/targets, including axial vectors."""
    x = image
    transformed = {k: v for k, v in targets.items()}

    if hflip:
        x = np.flip(x, axis=-1)
        transformed = {k: np.flip(v, axis=-1) for k, v in transformed.items()}
        if "orientation" in transformed:
            transformed["orientation"] = transformed["orientation"].copy()
            transformed["orientation"][1] *= -1.0

    if vflip:
        x = np.flip(x, axis=-2)
        transformed = {k: np.flip(v, axis=-2) for k, v in transformed.items()}
        if "orientation" in transformed:
            transformed["orientation"] = transformed["orientation"].copy()
            transformed["orientation"][1] *= -1.0

    return x.copy(), {k: v.copy() for k, v in transformed.items()}


class FilamentDataset(Dataset):
    def __init__(self, records, image_dir, patch_size=1024, train=True):
        self.records = records
        self.image_dir = Path(image_dir)
        self.patch_size = patch_size
        self.train = train

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = cv2.imread(str(self.image_dir / r["file_name"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(self.image_dir / r["file_name"])

        x = build_input_channels(img)
        t = build_targets(r)
        h, w = t["region"].shape
        s = min(self.patch_size, h, w)

        if self.train:
            if t["region"].any() and random.random() < 0.7:
                ys, xs = np.where(t["region"] > 0)
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
        cropped = {k: t[k][..., y : y + s, z : z + s] for k in TARGET_KEYS}

        if self.train:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            x, cropped = apply_geometric_transform(
                x,
                cropped,
                hflip=hflip,
                vflip=vflip,
            )

        out = {"image": torch.from_numpy(x).float()}
        for k in TARGET_KEYS:
            value = cropped[k]
            if k == "orientation":
                out[k] = torch.from_numpy(value.copy()).float()
            else:
                out[k] = torch.from_numpy(value[None].copy()).float()
        return out
