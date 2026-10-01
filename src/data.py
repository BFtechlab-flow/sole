from collections import defaultdict
from pathlib import Path
import json
import random
import warnings

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from scipy.ndimage import convolve, distance_transform_edt, gaussian_filter
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
    "width",
    "width_valid",
    "curvature",
    "curvature_valid",
    "endpoint",
    "junction",
    "instance_id",
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
    """Return axial tangent field [cos(2theta), sin(2theta)] on valid skeleton pixels."""
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


def skeleton_keypoints(skeleton, dilation_radius=2):
    """Innovations #6/#7: endpoint and junction targets from skeleton graph degree."""
    sk = np.asarray(skeleton, dtype=bool)
    neighbors = convolve(
        sk.astype(np.uint8),
        np.ones((3, 3), np.uint8),
        mode="constant",
        cval=0,
    ) - sk.astype(np.uint8)
    endpoint = sk & (neighbors == 1)
    junction = sk & (neighbors >= 3)
    if dilation_radius > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dilation_radius + 1, 2 * dilation_radius + 1),
        )
        endpoint = cv2.dilate(endpoint.astype(np.uint8), k) > 0
        junction = cv2.dilate(junction.astype(np.uint8), k) > 0
    return endpoint.astype(np.float32), junction.astype(np.float32)


def curvature_from_orientation(orientation, valid, sigma=2.0):
    """Innovation #5: normalized local tangent-change magnitude on the skeleton."""
    mask = valid.astype(np.float32)
    if not mask.any():
        return np.zeros_like(mask), mask
    smooth = np.zeros_like(orientation)
    denom = gaussian_filter(mask, sigma=sigma) + 1e-6
    for c in range(2):
        smooth[c] = gaussian_filter(orientation[c] * mask, sigma=sigma) / denom
    norm = np.linalg.norm(smooth, axis=0)
    good = (mask > 0) & (norm > 1e-6)
    smooth[:, good] /= norm[good]
    dot = np.clip(np.sum(orientation * smooth, axis=0), -1.0, 1.0)
    curvature = np.zeros_like(mask)
    curvature[good] = np.arccos(dot[good]) / np.pi
    return curvature.astype(np.float32), good.astype(np.float32)


def build_targets(r, width_scale_px=64.0):
    h, w = r["height"], r["width"]
    region = np.zeros((h, w), np.uint8)
    center = region.copy()
    boundary = region.copy()
    distance = np.zeros((h, w), np.float32)
    orientation_sum = np.zeros((2, h, w), np.float32)
    orientation_count = np.zeros((h, w), np.float32)
    width = np.zeros((h, w), np.float32)
    width_valid = np.zeros((h, w), np.float32)
    curvature = np.zeros((h, w), np.float32)
    curvature_valid = np.zeros((h, w), np.float32)
    endpoint = np.zeros((h, w), np.float32)
    junction = np.zeros((h, w), np.float32)
    instance_id = np.zeros((h, w), np.int32)
    instances = []
    kernel = np.ones((3, 3), np.uint8)

    for a in r["annotations"]:
        # Only segmentation GT is consumed. All advanced targets are derived from it.
        m = segmentation_to_mask(a["segmentation"], h, w)
        if not m.any():
            continue
        instances.append(m)
        current_id = len(instances)
        # COCO panoptic-style instance supervision should be non-overlapping. If
        # annotations overlap, preserve the first assignment so one pixel never
        # receives contradictory embedding labels.
        assign = (m > 0) & (instance_id == 0)
        instance_id[assign] = current_id
        region = np.maximum(region, m)

        skel = skeletonize(m > 0)
        skel_u8 = skel.astype(np.uint8)
        center = np.maximum(center, cv2.dilate(skel_u8, kernel, iterations=1))
        boundary = np.maximum(boundary, cv2.morphologyEx(m, cv2.MORPH_GRADIENT, kernel))

        d_raw = distance_transform_edt(m)
        d_norm = d_raw / (d_raw.max() + 1e-8)
        distance = np.maximum(distance, d_norm)

        ori, valid = orientation_from_skeleton(skel)
        orientation_sum += ori * valid[None]
        orientation_count += valid

        width_instance = np.clip((2.0 * d_raw) / max(float(width_scale_px), 1e-6), 0.0, 1.0)
        width_band = cv2.dilate(skel_u8, kernel, iterations=1).astype(bool)
        width[width_band] = np.maximum(width[width_band], width_instance[width_band])
        width_valid[width_band] = 1.0

        curv, curv_valid = curvature_from_orientation(ori, valid)
        curvature = np.maximum(curvature, curv)
        curvature_valid = np.maximum(curvature_valid, curv_valid)

        ep, jn = skeleton_keypoints(skel)
        endpoint = np.maximum(endpoint, ep)
        junction = np.maximum(junction, jn)

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
        "width": width.astype(np.float32),
        "width_valid": width_valid.astype(np.float32),
        "curvature": curvature.astype(np.float32),
        "curvature_valid": curvature_valid.astype(np.float32),
        "endpoint": endpoint.astype(np.float32),
        "junction": junction.astype(np.float32),
        "instance_id": instance_id,
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
    def __init__(
        self,
        records,
        image_dir,
        patch_size=1024,
        train=True,
        width_scale_px=64.0,
    ):
        self.records = records
        self.image_dir = Path(image_dir)
        self.patch_size = patch_size
        self.train = train
        self.width_scale_px = float(width_scale_px)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = cv2.imread(str(self.image_dir / r["file_name"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(self.image_dir / r["file_name"])

        x = build_input_channels(img)
        t = build_targets(r, width_scale_px=self.width_scale_px)
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
            elif k == "instance_id":
                out[k] = torch.from_numpy(value[None].copy()).long()
            else:
                out[k] = torch.from_numpy(value[None].copy()).float()
        return out
