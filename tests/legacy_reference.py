"""Frozen copy of the V3 (commit e7b19af) implementations that were optimised.

Used only by tests/test_fast_equivalence.py to prove that the fast versions in
src/ produce exactly the same outputs. Do not import from production code.
"""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from src.data import (
    curvature_from_orientation,
    orientation_from_skeleton,
    segmentation_to_mask,
    skeleton_keypoints,
)


def legacy_build_targets(r, width_scale_px=64.0):
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


