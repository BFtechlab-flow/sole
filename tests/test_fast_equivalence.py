"""The optimised code paths must give exactly the V3 results.

* ``src.data.build_targets`` (per-instance padded crops) vs the frozen V3
  whole-image implementation in ``tests/legacy_reference.py``;
* window mode == whole-image targets cropped to the window;
* ``src.reconstruct.reconstruct_instances`` with its default (legacy) options
  vs the frozen V3 module in ``tests/legacy_reconstruct.py``.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi

from src.data import TARGET_KEYS, build_targets
from src.reconstruct import reconstruct_instances

from tests.legacy_reconstruct import reconstruct_instances as legacy_reconstruct_instances
from tests.legacy_reference import legacy_build_targets


def random_polygon(rng, h, w, near_border=False):
    if near_border:
        cy = float(rng.choice([rng.uniform(0, 12), rng.uniform(h - 12, h)]))
        cx = float(rng.uniform(0, w))
    else:
        cy, cx = float(rng.uniform(20, h - 20)), float(rng.uniform(20, w - 20))
    n = int(rng.integers(5, 12))
    length = rng.uniform(20, 60)
    ang = rng.uniform(0, np.pi)
    t = np.linspace(-0.5, 0.5, n)
    spine = np.stack([cx + t * length * np.cos(ang), cy + t * length * np.sin(ang)], 1)
    spine += rng.normal(0, 3, spine.shape)
    half = rng.uniform(1.5, 6.0)
    normal = np.array([-np.sin(ang), np.cos(ang)])
    left = spine + half * normal
    right = spine[::-1] - half * normal
    poly = np.concatenate([left, right, left[:1]], 0)
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
    return poly.ravel().tolist()


def random_record(rng, h=150, w=190, n=None):
    n = int(rng.integers(1, 7)) if n is None else n
    anns = [{"segmentation": [random_polygon(rng, h, w, near_border=rng.random() < 0.25)]} for _ in range(n)]
    return {"height": h, "width": w, "annotations": anns}


@pytest.mark.parametrize("seed", range(8))
def test_fast_targets_identical_to_v3(seed):
    rng = np.random.default_rng(seed)
    record = random_record(rng)
    old = legacy_build_targets(record)
    new = build_targets(record)
    for key in TARGET_KEYS:
        assert old[key].dtype == new[key].dtype, key
        assert np.array_equal(old[key], new[key]), key
    assert len(old["instances"]) == len(new["instances"])
    for a, b in zip(old["instances"], new["instances"]):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("seed", range(4))
def test_window_targets_equal_cropped_full_targets(seed):
    rng = np.random.default_rng(100 + seed)
    record = random_record(rng)
    full = build_targets(record)
    for _ in range(3):
        wh, ww = int(rng.integers(30, 120)), int(rng.integers(30, 150))
        y0, x0 = int(rng.integers(0, 150 - wh + 1)), int(rng.integers(0, 190 - ww + 1))
        win = build_targets(record, window=(y0, x0, wh, ww))
        for key in TARGET_KEYS:
            ref = full[key][..., y0 : y0 + wh, x0 : x0 + ww]
            assert np.array_equal(ref, win[key]), key


def synthetic_fields(rng, h=256, w=288):
    region = np.zeros((h, w), np.float32)
    center = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(3, 8))):
        m = np.zeros((h, w), bool)
        y, x = int(rng.integers(20, h - 20)), int(rng.integers(20, w - 20))
        ang = rng.uniform(0, 2 * np.pi)
        for _ in range(int(rng.integers(15, 60))):
            ang += rng.normal(0, 0.2)
            y = int(np.clip(y + 2 * np.sin(ang), 0, h - 1))
            x = int(np.clip(x + 2 * np.cos(ang), 0, w - 1))
            m[y, x] = True
        c = ndi.binary_dilation(m, iterations=1)
        r = ndi.binary_dilation(m, iterations=int(rng.integers(2, 6)))
        if rng.random() < 0.5:  # break the centerline to exercise the bridging
            ys, xs = np.nonzero(c)
            k = len(ys) // 2
            c[max(0, ys[k] - 4) : ys[k] + 5, max(0, xs[k] - 4) : xs[k] + 5] = False
        region = np.maximum(region, r * rng.uniform(0.5, 0.95))
        center = np.maximum(center, c * rng.uniform(0.4, 0.95))
    region = np.clip(region + rng.normal(0, 0.05, region.shape), 0, 1).astype(np.float32)
    center = np.clip(center + rng.normal(0, 0.05, center.shape), 0, 1).astype(np.float32)
    distance = ndi.gaussian_filter(region, 2).astype(np.float32)
    boundary = np.clip(ndi.gaussian_gradient_magnitude(region, 1.5) * 2, 0, 1).astype(np.float32)
    orientation = rng.normal(size=(2, h, w)).astype(np.float32)
    orientation /= np.linalg.norm(orientation, axis=0, keepdims=True) + 1e-6
    width = (ndi.gaussian_filter(region, 3) * 0.3).astype(np.float32)
    return region, center, boundary, distance, orientation, width


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("geodesic", [False, True])
def test_fast_reconstruction_identical_to_v3(seed, geodesic):
    rng = np.random.default_rng(seed)
    region, center, boundary, distance, orientation, width = synthetic_fields(rng)
    kwargs = dict(
        orientation=orientation,
        width=width,
        region_threshold=0.45,
        center_threshold=0.35,
        min_region_area=16,
        min_instance_area=30,
        bridge_max_distance=20,
        bridge_min_alignment=0.5,
        bridge_geodesic=geodesic,
    )
    old = legacy_reconstruct_instances(region, center, boundary, distance, **kwargs)
    new = reconstruct_instances(region, center, boundary, distance, **kwargs)
    assert len(old) == len(new)
    for a, b in zip(old, new):
        assert np.array_equal(a, b)
