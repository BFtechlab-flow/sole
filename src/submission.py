from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .rle import decode_counts, encode_counts


REQUIRED_COLUMNS = ["filament_id", "segmentation_rle"]


def _match_image_stem(filament_id: str, stems):
    matches = [stem for stem in stems if filament_id.startswith(stem + "_")]
    if not matches:
        return None
    return max(matches, key=len)


def validate_submission_frame(frame: pd.DataFrame, image_shapes: dict[str, tuple[int, int]], require_nonoverlap: bool = True) -> dict:
    errors = []
    warnings = []
    decoded_by_image = {stem: [] for stem in image_shapes}

    if list(frame.columns) != REQUIRED_COLUMNS:
        errors.append(f"columns must be exactly {REQUIRED_COLUMNS}, got {list(frame.columns)}")
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing_columns:
        errors.append(f"missing required columns: {missing_columns}")
        return {
            "ok": False,
            "rows": int(len(frame)),
            "images_with_predictions": 0,
            "total_images": int(len(image_shapes)),
            "overlap_pixels": 0,
            "errors": errors,
            "warnings": warnings,
        }

    if frame["filament_id"].isna().any() or frame["segmentation_rle"].isna().any():
        errors.append("submission contains missing filament_id or segmentation_rle")
    if frame["filament_id"].duplicated().any():
        errors.append("filament_id values must be unique")

    seen_rle = set()
    for row_idx, row in frame.iterrows():
        fid = str(row["filament_id"])
        rle = str(row["segmentation_rle"])
        stem = _match_image_stem(fid, image_shapes.keys())
        if stem is None:
            errors.append(f"row {row_idx}: unknown image prefix in filament_id={fid!r}")
            continue
        h, w = image_shapes[stem]
        try:
            mask = decode_counts(rle, h, w)
        except Exception as exc:
            errors.append(f"row {row_idx}: RLE decode failed: {exc}")
            continue
        if mask.shape != (h, w):
            errors.append(f"row {row_idx}: decoded shape {mask.shape} != {(h, w)}")
            continue
        if not np.any(mask):
            errors.append(f"row {row_idx}: empty instance mask")
        if rle in seen_rle:
            warnings.append(f"row {row_idx}: duplicate compressed RLE payload")
        seen_rle.add(rle)
        try:
            canonical = encode_counts(mask)
            roundtrip = decode_counts(canonical, h, w)
            if not np.array_equal(mask.astype(np.uint8), roundtrip.astype(np.uint8)):
                errors.append(f"row {row_idx}: RLE round-trip mismatch")
        except Exception as exc:
            errors.append(f"row {row_idx}: RLE round-trip failed: {exc}")
        decoded_by_image[stem].append(mask.astype(bool))

    overlap_pixels = 0
    if require_nonoverlap:
        for stem, masks in decoded_by_image.items():
            if len(masks) < 2:
                continue
            occupancy = np.zeros(image_shapes[stem], dtype=np.uint16)
            for mask in masks:
                occupancy += mask
            overlap = int(np.sum(occupancy > 1))
            overlap_pixels += overlap
            if overlap:
                errors.append(f"{stem}: {overlap} pixels belong to more than one predicted instance")

    return {
        "ok": not errors,
        "rows": int(len(frame)),
        "images_with_predictions": int(sum(bool(v) for v in decoded_by_image.values())),
        "total_images": int(len(image_shapes)),
        "overlap_pixels": int(overlap_pixels),
        "errors": errors,
        "warnings": warnings,
    }


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_verification(report: dict, submission_path, output_path):
    payload = dict(report)
    payload["submission"] = str(submission_path)
    payload["sha256"] = sha256_file(submission_path)
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
