"""Shared evaluation helpers: GT/prediction RLEs and per-image official scoring.

Everything that turns predictions into competition numbers goes through here,
so training-time validation, OOF evaluation, post-processing search and the
ablations all use exactly the official metric (``src.pq_official``).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np
from pycocotools import mask as mask_utils
from scipy.ndimage import find_objects

from .pq_official import aggregate_scores, pq_score
from .utils import observation_key


def annotation_rle(segmentation, h: int, w: int) -> dict:
    """COCO annotation -> RLE exactly like ``COCO.annToRLE`` (used by the organizers)."""
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, h, w)
        return mask_utils.merge(rles)
    if isinstance(segmentation.get("counts"), list):
        return mask_utils.frPyObjects(segmentation, h, w)
    rle = dict(segmentation)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("ascii")
    return rle


def record_gt_rles(record) -> list[dict]:
    """All GT filaments of one annotator record (empty ones included, as in the official GT)."""
    h, w = int(record["height"]), int(record["width"])
    return [annotation_rle(a["segmentation"], h, w) for a in record["annotations"]]


def label_map_to_rles(labels: np.ndarray) -> list[dict]:
    """One RLE per instance label (1..N), built from bounding-box crops.

    Masks are written straight into Fortran-ordered buffers, which avoids the
    full-image copies/transpositions that dominate ``encode`` otherwise.
    """
    h, w = labels.shape
    out = []
    for i, sl in enumerate(find_objects(labels), start=1):
        buf = np.zeros((h, w), dtype=np.uint8, order="F")
        if sl is not None:
            buf[sl] = labels[sl] == i
        out.append(mask_utils.encode(buf))
    return out


def masks_to_rles(masks: Iterable[np.ndarray]) -> list[dict]:
    return [mask_utils.encode(np.asfortranarray((np.asarray(m) > 0).astype(np.uint8))) for m in masks]


def score_image(records: Sequence[dict], pred_rles: Sequence[dict], threshold: float = 0.5,
                relation_threshold: float = 0.10) -> list[dict]:
    """Score one prediction set against every annotator record of the same image."""
    shape = None
    scores = []
    for record in records:
        gt = record_gt_rles(record)
        if shape is None:
            shape = (int(record["height"]), int(record["width"]))
        s = pq_score(gt, list(pred_rles), threshold=threshold, relation_threshold=relation_threshold, shape=shape)
        s["entry"] = str(record["image_id"])
        s["stem"] = observation_key(record["file_name"])
        scores.append(s)
    return scores


def group_records_by_image(records: Sequence[dict], indices: Iterable[int] | None = None) -> dict[str, list[int]]:
    """file_name -> indices of its annotator records (in input order)."""
    by_file: dict[str, list[int]] = defaultdict(list)
    for i in (range(len(records)) if indices is None else indices):
        by_file[records[i]["file_name"]].append(int(i))
    return dict(by_file)


def summarize(scores: Sequence[dict]) -> dict:
    """Official aggregate + the observation-level grouping used by bootstraps."""
    out = aggregate_scores(scores)
    out["observations"] = int(len({s.get("stem") for s in scores}))
    return out


def subsample_images(files: Sequence[str], max_images: int | None, seed: int = 2026) -> list[str]:
    """Deterministic subset of image names (for quick validation passes)."""
    files = sorted(files)
    if max_images is None or max_images <= 0 or max_images >= len(files):
        return files
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(files), size=int(max_images), replace=False))
    return [files[i] for i in idx]
