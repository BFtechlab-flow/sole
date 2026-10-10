"""Submission CSV: writing and strict validation (competition format).

Format (Overview > Submission File): one CSV for the whole test set, header
``filament_id,segmentation_rle``, one row per predicted filament,
``filament_id = <image stem>_<k>``, ``segmentation_rle`` = pycocotools
compressed RLE *counts* of a 2048x2048 mask, without size and without quotes.
The scorer takes the image id as the part of ``filament_id`` before the first
``_``, so that is how ids are parsed here too.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .rle import decode_counts, encode_counts

REQUIRED_COLUMNS = ["filament_id", "segmentation_rle"]
COMPETITION_SHAPE = (2048, 2048)


def rle_string_to_runs(s: str) -> list[int]:
    """Python port of pycocotools ``rleFrString`` used to validate an RLE string.

    pycocotools' C decoder performs no bounds checking, so malformed strings
    are checked here before they are decoded.
    """
    runs: list[int] = []
    p, n = 0, len(s)
    while p < n:
        x = 0
        k = 0
        more = True
        while more:
            if p >= n:
                raise ValueError("truncated RLE string")
            c = ord(s[p]) - 48
            if c < 0 or c > 63:
                raise ValueError(f"invalid RLE character {s[p]!r}")
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(runs) > 2:
            x += runs[-2]
        runs.append(x)
    return runs


def image_id_of(filament_id: str) -> str:
    return str(filament_id).split("_", 1)[0]


def write_submission(frame: pd.DataFrame, path) -> Path:
    """Write exactly two columns, header, no index, no quoting (fails loudly if quoting were needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, REQUIRED_COLUMNS].astype(str).to_csv(
        path, index=False, quoting=csv.QUOTE_NONE, lineterminator="\n"
    )
    return path


def validate_submission_frame(
    frame: pd.DataFrame,
    image_shapes: dict[str, tuple[int, int]],
    require_nonoverlap: bool = True,
    expected_shape: tuple[int, int] | None = COMPETITION_SHAPE,
) -> dict:
    """Validate a submission frame against the test image ids/shapes.

    Memory use is bounded by one image at a time (masks are decoded per image
    and accumulated into a single occupancy map).
    """
    errors: list[str] = []
    warnings: list[str] = []

    def report(**extra):
        base = {
            "ok": not errors,
            "rows": int(len(frame)),
            "images_with_predictions": 0,
            "total_images": int(len(image_shapes)),
            "overlap_pixels": 0,
            "errors": errors,
            "warnings": warnings,
        }
        base.update(extra)
        base["ok"] = not errors
        return base

    if list(frame.columns) != REQUIRED_COLUMNS:
        errors.append(f"columns must be exactly {REQUIRED_COLUMNS}, got {list(frame.columns)}")
    if any(c not in frame.columns for c in REQUIRED_COLUMNS):
        return report()

    ids = frame["filament_id"]
    rles = frame["segmentation_rle"]
    if ids.isna().any() or rles.isna().any() or (ids.astype(str) == "").any() or (rles.astype(str) == "").any():
        errors.append("submission contains missing/empty filament_id or segmentation_rle values")
    if ids.astype(str).duplicated().any():
        errors.append("filament_id values must be unique")
    if not ids.astype(str).str.contains("_", regex=False).all():
        errors.append("every filament_id must be '<image id>_<k>'")
    if expected_shape is not None:
        wrong = {k: v for k, v in image_shapes.items() if tuple(v) != tuple(expected_shape)}
        if wrong:
            errors.append(f"{len(wrong)} test images are not {expected_shape}, e.g. {next(iter(wrong.items()))}")

    rows_by_image: dict[str, list[int]] = defaultdict(list)
    for row_idx, fid in enumerate(ids.astype(str)):
        stem = image_id_of(fid)
        if stem not in image_shapes:
            errors.append(f"row {row_idx}: image id {stem!r} of filament_id={fid!r} is not a test image")
            continue
        rows_by_image[stem].append(row_idx)

    seen_rle = set()
    overlap_pixels = 0
    areas = []
    for stem, row_indices in rows_by_image.items():
        h, w = image_shapes[stem]
        occupancy = np.zeros((h, w), dtype=np.uint16) if require_nonoverlap else None
        for row_idx in row_indices:
            rle = str(rles.iat[row_idx])
            try:
                runs = rle_string_to_runs(rle)
                if any(r < 0 for r in runs) or sum(runs) != h * w:
                    raise ValueError(f"RLE does not describe a {h}x{w} mask")
                mask = decode_counts(rle, h, w)
            except Exception as exc:  # noqa: BLE001 - report every malformed row
                errors.append(f"row {row_idx}: invalid RLE: {exc}")
                continue
            area = int(mask.sum())
            areas.append(area)
            if area == 0:
                errors.append(f"row {row_idx}: empty instance mask")
            if rle in seen_rle:
                warnings.append(f"row {row_idx}: duplicate compressed RLE payload")
            seen_rle.add(rle)
            if encode_counts(mask) != rle:
                warnings.append(f"row {row_idx}: RLE is valid but not in canonical pycocotools form")
            if occupancy is not None:
                occupancy += mask
        if occupancy is not None:
            overlap = int(np.count_nonzero(occupancy > 1))
            overlap_pixels += overlap
            if overlap:
                errors.append(f"{stem}: {overlap} pixels belong to more than one predicted instance")

    missing = sorted(set(image_shapes) - set(rows_by_image))
    if missing:
        warnings.append(f"{len(missing)} test images have no predicted filament (all their GT filaments become FN)")
    stats = {}
    if areas:
        a = np.asarray(areas)
        stats = {"area_min": int(a.min()), "area_median": float(np.median(a)), "area_max": int(a.max())}
    per_image = [len(v) for v in rows_by_image.values()]
    return report(
        images_with_predictions=int(len(rows_by_image)),
        overlap_pixels=int(overlap_pixels),
        filaments_per_image_mean=float(np.mean(per_image)) if per_image else 0.0,
        mask_area=stats,
    )


def validate_submission_file(path, image_shapes, require_nonoverlap=True,
                             expected_shape=COMPETITION_SHAPE) -> dict:
    """File-level checks (header, quotes) + :func:`validate_submission_frame`."""
    raw = Path(path).read_text(encoding="utf-8")
    first = raw.split("\n", 1)[0].rstrip("\r")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    rep = validate_submission_frame(frame, image_shapes, require_nonoverlap=require_nonoverlap,
                                    expected_shape=expected_shape)
    if first != ",".join(REQUIRED_COLUMNS):
        rep["errors"].append(f"header must be exactly {','.join(REQUIRED_COLUMNS)!r}, got {first!r}")
    if '"' in raw or "'" in raw:
        rep["errors"].append("the file contains quote characters; RLE strings must not be quoted")
    rep["ok"] = not rep["errors"]
    return rep


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
