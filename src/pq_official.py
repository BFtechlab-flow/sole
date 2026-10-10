"""Panoptic Quality exactly as scored by the competition, plus rubric diagnostics.

Reference: the organizers' public *Self_Evaluation_Notebook* (Kaggle,
version 6, 9 Aug 2026), which the host described as the exact implementation
of the leaderboard PQ. ``tests/test_official_parity.py`` checks this module
against an independent dense re-implementation of the same algorithm
(``tests/reference_pq_dense.py``), and
``tools/verify_against_official_notebook.py`` runs the organizers' notebook
itself (downloaded by the user, not redistributed) on the same inputs.

Official semantics reproduced here
----------------------------------
* The unit of scoring is an **annotator-image entry**: one annotation set of
  one observation (``<batch>-<stem>`` in the training JSON). Observations
  annotated by 2-3 experts contribute 2-3 entries, each compared with the same
  prediction set of that image.
* Within an entry a (GT, prediction) pair is a hit when ``IoU > 0.5``
  (strict). Every hit is a TP and contributes its IoU. A prediction without
  any hit is an FP; a GT filament without any hit is an FN. There is no
  one-to-one assignment step: if overlapping predictions both exceed 0.5 with
  the same GT, both count (this cannot happen with disjoint predictions,
  which is what our pipeline produces).
* Counts are summed over **all** entries and PQ is computed once (micro)::

      PQ = sum(IoU of TPs) / (|TP| + 0.5 |FP| + 0.5 |FN|)

* Only entries with at least one GT filament exist in the official ground
  truth (it is built from the GT rows), so entries without GT filaments are
  not scored (predictions on them are free).

IoUs are computed on COCO run-length encodings with pycocotools and then
re-derived from the integer intersection/union counts in float32, which is
bit-identical to the notebook's dense float32 computation.

The macro average of per-entry PQ is kept only as a diagnostic: it is NOT the
competition score and must not drive model selection.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np
from pycocotools import mask as mask_utils

IOU_THRESHOLD = 0.50
#: IoU > 0 is what the organizers' notebook uses for its one-to-many /
#: many-to-one plot (part of the judging rubric).
RELATION_THRESHOLD_OFFICIAL = 0.0


# --------------------------------------------------------------------------- RLE helpers
def to_rle(item, shape: Sequence[int] | None = None) -> dict:
    """Convert a binary mask, an RLE dict or an RLE counts string to an RLE dict."""
    if isinstance(item, dict):
        rle = dict(item)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("ascii")
        return rle
    if isinstance(item, (str, bytes)):
        if shape is None:
            raise ValueError("shape is required to interpret an RLE counts string")
        counts = item.encode("ascii") if isinstance(item, str) else item
        return {"size": [int(shape[0]), int(shape[1])], "counts": counts}
    arr = np.asarray(item)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {arr.shape}")
    return mask_utils.encode(np.asfortranarray((arr > 0).astype(np.uint8)))


def to_rles(items: Iterable, shape: Sequence[int] | None = None) -> list[dict]:
    return [to_rle(x, shape) for x in items]


def _shape_of(rles: list[dict]) -> tuple[int, int] | None:
    for r in rles:
        return int(r["size"][0]), int(r["size"][1])
    return None


def iou_matrix(gt: Sequence, pred: Sequence, shape: Sequence[int] | None = None) -> np.ndarray:
    """Pairwise IoU (rows = GT, columns = predictions), float32 like the official code."""
    gt_r = to_rles(gt, shape)
    pred_r = to_rles(pred, shape)
    n_gt, n_pred = len(gt_r), len(pred_r)
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred), dtype=np.float32)
    s_gt, s_pred = _shape_of(gt_r), _shape_of(pred_r)
    if s_gt != s_pred:
        raise ValueError(f"GT masks have shape {s_gt} but predictions have shape {s_pred}")
    # pycocotools: iou(dt, gt, iscrowd) -> (n_dt, n_gt), computed in double as I / U
    iou64 = np.asarray(mask_utils.iou(pred_r, gt_r, [0] * n_gt), dtype=np.float64).reshape(n_pred, n_gt).T
    area_gt = mask_utils.area(gt_r).astype(np.float64)[:, None]
    area_pred = mask_utils.area(pred_r).astype(np.float64)[None, :]
    # Recover the exact integer intersection, then divide in float32 exactly as
    # torch does in the notebook (I and U are exact integers in float32).
    inter = np.rint(iou64 * (area_gt + area_pred) / (1.0 + iou64))
    union = area_gt + area_pred - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou32 = np.where(union > 0, inter.astype(np.float32) / union.astype(np.float32), np.float32(0.0))
    return iou32.astype(np.float32)


def iou_to_dice(iou: np.ndarray) -> np.ndarray:
    iou = np.asarray(iou, dtype=np.float64)
    return np.where(iou > 0, 2.0 * iou / (1.0 + iou), 0.0)


# --------------------------------------------------------------------------- per entry
def pq_score(
    gt: Sequence,
    pred: Sequence,
    threshold: float = IOU_THRESHOLD,
    relation_threshold: float = 0.10,
    shape: Sequence[int] | None = None,
    enforce_unique_matches: bool | None = None,  # deprecated, kept for API compatibility
) -> dict:
    """Score ONE annotator-image entry against the prediction set of its image.

    ``gt`` / ``pred`` may contain binary masks, RLE dicts or counts strings
    (then ``shape`` is required). The returned dict holds the counts that the
    official micro aggregation sums (``tp``, ``fp``, ``fn``, ``iou_sum``) and
    the rubric diagnostics (IoU/Dice distributions, 1:n and n:1 relations).
    """
    del enforce_unique_matches
    iou = iou_matrix(gt, pred, shape)
    n_gt, n_pred = iou.shape
    if n_gt and n_pred:
        hit = iou > float(threshold)
        matched = iou[hit].astype(np.float64)
        tp = int(hit.sum())
        fp = int((hit.sum(axis=0) == 0).sum())
        fn = int((hit.sum(axis=1) == 0).sum())
        touch = iou > RELATION_THRESHOLD_OFFICIAL
        gt_degree = touch.sum(axis=1).astype(int)
        pred_degree = touch.sum(axis=0).astype(int)
        overlap = iou[touch].astype(np.float64)
        rel = iou > float(relation_threshold)
        rel_gt, rel_pred = rel.sum(axis=1), rel.sum(axis=0)
        pairs = [(int(i), int(j), float(iou[i, j])) for i, j in np.argwhere(hit)]
    else:
        matched = np.zeros(0)
        tp, fp, fn = 0, n_pred, n_gt
        gt_degree = np.zeros(n_gt, dtype=int)
        pred_degree = np.zeros(n_pred, dtype=int)
        overlap = np.zeros(0)
        rel_gt, rel_pred = gt_degree, pred_degree
        pairs = []

    iou_sum = float(matched.sum())
    denominator = tp + 0.5 * fp + 0.5 * fn
    scored = n_gt > 0
    return {
        "scored": bool(scored),
        "pq": (iou_sum / denominator if denominator else 0.0) if scored else None,
        "sq": (iou_sum / tp if tp else 0.0) if scored else None,
        "rq": (tp / denominator if denominator else 0.0) if scored else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou_sum": iou_sum,
        "pairs": pairs,
        "matched_ious": matched.tolist(),
        "matched_dices": iou_to_dice(matched).tolist(),
        "overlap_ious": overlap.tolist(),
        "overlap_dices": iou_to_dice(overlap).tolist(),
        "gt_degree": gt_degree.tolist(),
        "pred_degree": pred_degree.tolist(),
        "one_to_many": int(np.sum(gt_degree > 1)),
        "many_to_one": int(np.sum(pred_degree > 1)),
        "relation_threshold": float(relation_threshold),
        "one_to_many_at_threshold": int(np.sum(np.asarray(rel_gt) > 1)),
        "many_to_one_at_threshold": int(np.sum(np.asarray(rel_pred) > 1)),
        "gt_count": int(n_gt),
        "pred_count": int(n_pred),
    }


# --------------------------------------------------------------------------- aggregation
def _distribution(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        keys = ("mean", "std", "min", "q10", "q25", "median", "q75", "q90", "max")
        return {"count": 0, **{k: None for k in keys}}
    q = np.quantile(arr, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "q10": float(q[0]),
        "q25": float(q[1]),
        "median": float(q[2]),
        "q75": float(q[3]),
        "q90": float(q[4]),
        "max": float(arr.max()),
    }


def micro_pq(scores: Sequence[dict]) -> float:
    """The competition score: global sums over the scored entries."""
    sel = [s for s in scores if s.get("scored", s.get("gt_count", 1) > 0)]
    tp = sum(s["tp"] for s in sel)
    fp = sum(s["fp"] for s in sel)
    fn = sum(s["fn"] for s in sel)
    den = tp + 0.5 * fp + 0.5 * fn
    return float(sum(s["iou_sum"] for s in sel) / den) if den > 0 else 0.0


def aggregate_scores(scores: Sequence[dict]) -> dict:
    """Aggregate per-entry scores. ``pq``/``micro_pq`` is the official score."""
    sel = [s for s in scores if s.get("scored", s.get("gt_count", 1) > 0)]
    tp = int(sum(s["tp"] for s in sel))
    fp = int(sum(s["fp"] for s in sel))
    fn = int(sum(s["fn"] for s in sel))
    iou_sum = float(sum(s["iou_sum"] for s in sel))
    den = tp + 0.5 * fp + 0.5 * fn
    pq = iou_sum / den if den > 0 else 0.0
    gt_deg = Counter(d for s in sel for d in s.get("gt_degree", []))
    pred_deg = Counter(d for s in sel for d in s.get("pred_degree", []))
    n_gt = sum(gt_deg.values())
    n_pred = sum(pred_deg.values())
    per_entry_pq = [s["pq"] for s in sel if s.get("pq") is not None]
    return {
        "pq": float(pq),
        "micro_pq": float(pq),
        "sq": float(iou_sum / tp) if tp else 0.0,
        "rq": float(tp / den) if den else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou_sum": iou_sum,
        "records": int(len(scores)),
        "scored_records": int(len(sel)),
        "unscored_records_without_gt": int(len(scores) - len(sel)),
        "gt_instances": int(n_gt),
        "pred_instance_evaluations": int(n_pred),
        "macro_pq_diagnostic": float(np.mean(per_entry_pq)) if per_entry_pq else 0.0,
        "matched_iou_distribution": _distribution(v for s in sel for v in s.get("matched_ious", [])),
        "matched_dice_distribution": _distribution(v for s in sel for v in s.get("matched_dices", [])),
        "overlap_iou_distribution": _distribution(v for s in sel for v in s.get("overlap_ious", [])),
        "overlap_dice_distribution": _distribution(v for s in sel for v in s.get("overlap_dices", [])),
        "gt_degree_histogram": {str(k): int(v) for k, v in sorted(gt_deg.items())},
        "pred_degree_histogram": {str(k): int(v) for k, v in sorted(pred_deg.items())},
        "one_to_many": int(sum(s.get("one_to_many", 0) for s in sel)),
        "many_to_one": int(sum(s.get("many_to_one", 0) for s in sel)),
        "fragmentation_rate": float(sum(v for k, v in gt_deg.items() if k >= 2) / n_gt) if n_gt else 0.0,
        "merge_rate": float(sum(v for k, v in pred_deg.items() if k >= 2) / n_pred) if n_pred else 0.0,
        "miss_rate": float(gt_deg.get(0, 0) / n_gt) if n_gt else 0.0,
        "spurious_rate": float(pred_deg.get(0, 0) / n_pred) if n_pred else 0.0,
    }


# --------------------------------------------------------------------------- uncertainty
def _group_sums(scores: Sequence[dict], groups: Sequence | None):
    sel_idx = [i for i, s in enumerate(scores) if s.get("scored", s.get("gt_count", 1) > 0)]
    if groups is None:
        keys = list(range(len(sel_idx)))
    else:
        keys = [groups[i] for i in sel_idx]
    uniq = {k: n for n, k in enumerate(dict.fromkeys(keys))}
    sums = np.zeros((len(uniq), 4), dtype=np.float64)  # iou_sum, tp, fp, fn
    for i, k in zip(sel_idx, keys):
        s = scores[i]
        sums[uniq[k]] += (s["iou_sum"], s["tp"], s["fp"], s["fn"])
    return sums, list(uniq)


def _pq_from_sums(sums: np.ndarray) -> np.ndarray:
    den = sums[..., 1] + 0.5 * sums[..., 2] + 0.5 * sums[..., 3]
    return np.where(den > 0, sums[..., 0] / np.maximum(den, 1e-12), 0.0)


def bootstrap_micro_pq(scores: Sequence[dict], groups: Sequence | None = None,
                       iterations: int = 2000, seed: int = 2026) -> dict:
    """Bootstrap CI of the micro PQ, resampling whole observations (``groups``)."""
    sums, keys = _group_sums(scores, groups)
    if len(keys) == 0:
        return {"iterations": 0, "pq": 0.0, "ci95": [0.0, 0.0], "groups": 0}
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(keys), np.full(len(keys), 1.0 / len(keys)), size=int(iterations))
    draws = _pq_from_sums(w @ sums)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "iterations": int(iterations),
        "pq": float(_pq_from_sums(sums.sum(axis=0))),
        "ci95": [float(lo), float(hi)],
        "std": float(draws.std()),
        "groups": int(len(keys)),
    }


def paired_bootstrap_micro_pq(scores_a: Sequence[dict], scores_b: Sequence[dict],
                              groups: Sequence, iterations: int = 5000, seed: int = 2026) -> dict:
    """Paired bootstrap of ``PQ(b) - PQ(a)`` on the same entries, resampling observations."""
    if len(scores_a) != len(scores_b) or len(scores_a) != len(groups):
        raise ValueError("scores_a, scores_b and groups must describe the same entries")
    sa, ka = _group_sums(scores_a, groups)
    sb, kb = _group_sums(scores_b, groups)
    if ka != kb:
        raise ValueError("the two experiments do not score the same set of entries")
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(ka), np.full(len(ka), 1.0 / len(ka)), size=int(iterations))
    delta = _pq_from_sums(w @ sb) - _pq_from_sums(w @ sa)
    lo, hi = np.quantile(delta, [0.025, 0.975])
    pa, pb = float(_pq_from_sums(sa.sum(0))), float(_pq_from_sums(sb.sum(0)))
    return {
        "baseline_pq": pa,
        "candidate_pq": pb,
        "delta_pq": pb - pa,
        "ci95": [float(lo), float(hi)],
        "probability_positive": float(np.mean(delta > 0)),
        "iterations": int(iterations),
        "groups": int(len(ka)),
    }


def bootstrap_macro_pq(scores: Sequence[dict], iterations: int = 1000, seed: int = 2026) -> dict:
    """Bootstrap of the macro (per-entry mean) PQ. Diagnostic only."""
    vals = np.asarray([s["pq"] for s in scores if s.get("pq") is not None], dtype=np.float64)
    if vals.size == 0:
        return {"iterations": 0, "mean": 0.0, "ci95": [0.0, 0.0]}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(int(iterations), vals.size))
    draws = vals[idx].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"iterations": int(iterations), "mean": float(vals.mean()), "ci95": [float(lo), float(hi)]}


# --------------------------------------------------------------------------- frames (submission format)
def evaluate_frames(gt_df, pred_df, shape: Sequence[int] = (2048, 2048), threshold: float = IOU_THRESHOLD) -> dict:
    """Official scoring of two frames in submission format.

    ``gt_df`` ids look like ``<batch>-<stem>_<k>``, ``pred_df`` ids like
    ``<stem>_<k>``; both have a ``segmentation_rle`` column of counts strings.
    Returns the aggregate plus the per-entry scores.
    """
    import pandas as pd

    for name, df in (("gt_df", gt_df), ("pred_df", pred_df)):
        missing = {"filament_id", "segmentation_rle"} - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing columns {sorted(missing)}")
    gt_entry = gt_df["filament_id"].astype(str).str.split("_", n=1).str[0]
    pred_img = (pred_df["filament_id"].astype(str).str.split("_", n=1).str[0]
                if len(pred_df) else pd.Series([], dtype=str))
    gt_groups = gt_df["segmentation_rle"].astype(str).groupby(gt_entry.values, sort=False)
    pred_groups = ({k: list(v) for k, v in pred_df["segmentation_rle"].astype(str).groupby(pred_img.values, sort=False)}
                   if len(pred_df) else {})
    scores, entries = [], []
    for entry, rles in gt_groups:
        if "-" not in entry:
            raise ValueError(f"ground-truth id {entry!r} is not '<annotator>-<image_id>_<k>'")
        image_id = entry.split("-", 1)[1]
        s = pq_score(list(rles), pred_groups.get(image_id, []), threshold=threshold, shape=shape)
        s["entry"], s["image_id"] = entry, image_id
        scores.append(s)
        entries.append(image_id)
    out = aggregate_scores(scores)
    out["per_entry"] = scores
    return out
