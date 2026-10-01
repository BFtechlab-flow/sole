"""Competition-aligned Panoptic Quality utilities for FILA-Net.

The organizer's public competition description defines a match as IoU strictly
greater than 0.5 and PQ as sum(matched IoU) / (TP + 0.5 FP + 0.5 FN).
MAGFiLO contains multiple independent annotation records for the same physical
JPEG; predictions must therefore be scored against every annotation record,
never against an annotator-union target.

The organizer's Self Evaluation notebook remains the final reference. This
module intentionally exposes both macro (mean record PQ) and micro
(global-count) aggregation so parity can be checked explicitly if the notebook
changes.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def pairwise_iou(gt: Sequence[np.ndarray], pred: Sequence[np.ndarray]) -> np.ndarray:
    """Return a dense GT x prediction IoU matrix."""
    matrix = np.zeros((len(gt), len(pred)), dtype=np.float32)
    for i, g in enumerate(gt):
        gb = np.asarray(g, dtype=bool)
        for j, p in enumerate(pred):
            pb = np.asarray(p, dtype=bool)
            inter = int(np.logical_and(gb, pb).sum())
            union = int(np.logical_or(gb, pb).sum())
            matrix[i, j] = inter / union if union else 0.0
    return matrix


def _relation_diagnostics(iou: np.ndarray, threshold: float = 0.10) -> dict:
    """Describe fragmentation/merging using a deliberately separate overlap threshold.

    This is a qualitative diagnostic, not part of PQ. A GT with >1 predictions
    above the relation threshold is one-to-many (fragmentation); a prediction
    overlapping >1 GT objects is many-to-one (merge).
    """
    if iou.size == 0:
        return {
            "relation_threshold": float(threshold),
            "one_to_many": 0,
            "many_to_one": 0,
            "gt_relation_degree": [],
            "pred_relation_degree": [],
        }
    linked = iou > float(threshold)
    gt_degree = linked.sum(axis=1).astype(int)
    pred_degree = linked.sum(axis=0).astype(int)
    return {
        "relation_threshold": float(threshold),
        "one_to_many": int(np.sum(gt_degree > 1)),
        "many_to_one": int(np.sum(pred_degree > 1)),
        "gt_relation_degree": gt_degree.tolist(),
        "pred_relation_degree": pred_degree.tolist(),
    }


def pq_score(
    gt: Sequence[np.ndarray],
    pred: Sequence[np.ndarray],
    threshold: float = 0.50,
    relation_threshold: float = 0.10,
    enforce_unique_matches: bool = True,
) -> dict:
    """Score one independent annotation record against one prediction set.

    The match threshold is *strictly greater than* ``threshold``. Under valid
    panoptic masks, IoU > 0.5 implies a unique match. If overlapping masks
    violate that property, ``enforce_unique_matches`` raises rather than
    silently changing the organizer semantics with a Hungarian matcher.
    """
    iou = pairwise_iou(gt, pred)
    match_idx = np.argwhere(iou > float(threshold))

    if len(match_idx):
        gt_ids = match_idx[:, 0]
        pred_ids = match_idx[:, 1]
        duplicate_gt = len(np.unique(gt_ids)) != len(gt_ids)
        duplicate_pred = len(np.unique(pred_ids)) != len(pred_ids)
        if enforce_unique_matches and (duplicate_gt or duplicate_pred):
            raise ValueError(
                "IoU>threshold produced non-unique matches. "
                "Predicted/GT instances likely overlap; official panoptic matching "
                "assumes uniqueness above 0.5."
            )

    pairs = [(int(i), int(j), float(iou[i, j])) for i, j in match_idx]
    tp = len(pairs)
    fp = len(pred) - tp
    fn = len(gt) - tp
    if fp < 0 or fn < 0:
        raise ValueError("Invalid PQ counts caused by non-unique matches.")

    ious = [v for _, _, v in pairs]
    dices = [(2.0 * v / (1.0 + v)) if v > 0 else 0.0 for v in ious]
    iou_sum = float(sum(ious))
    denominator = float(tp + 0.5 * fp + 0.5 * fn)

    pq = iou_sum / denominator if denominator else 1.0
    sq = iou_sum / tp if tp else (1.0 if denominator == 0 else 0.0)
    rq = tp / denominator if denominator else 1.0

    out = {
        "pq": float(pq),
        "sq": float(sq),
        "rq": float(rq),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "iou_sum": float(iou_sum),
        "pairs": pairs,
        "matched_ious": ious,
        "matched_dices": dices,
        "gt_count": int(len(gt)),
        "pred_count": int(len(pred)),
    }
    out.update(_relation_diagnostics(iou, relation_threshold))
    return out


def _distribution(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q10": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q90": None,
            "max": None,
        }
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


def aggregate_scores(scores: Sequence[dict]) -> dict:
    """Aggregate annotation-record scores for OOF/final diagnostics."""
    if not scores:
        return {
            "records": 0,
            "macro_pq": 0.0,
            "macro_sq": 0.0,
            "macro_rq": 0.0,
            "micro_pq": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "iou_distribution": _distribution([]),
            "dice_distribution": _distribution([]),
            "one_to_many": 0,
            "many_to_one": 0,
        }

    tp = int(sum(s["tp"] for s in scores))
    fp = int(sum(s["fp"] for s in scores))
    fn = int(sum(s["fn"] for s in scores))
    iou_sum = float(sum(s.get("iou_sum", sum(s.get("matched_ious", []))) for s in scores))
    denominator = tp + 0.5 * fp + 0.5 * fn

    ious = [v for s in scores for v in s.get("matched_ious", [])]
    dices = [v for s in scores for v in s.get("matched_dices", [])]
    return {
        "records": int(len(scores)),
        "macro_pq": float(np.mean([s["pq"] for s in scores])),
        "macro_sq": float(np.mean([s["sq"] for s in scores])),
        "macro_rq": float(np.mean([s["rq"] for s in scores])),
        "micro_pq": float(iou_sum / denominator if denominator else 1.0),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou_sum": iou_sum,
        "iou_distribution": _distribution(ious),
        "dice_distribution": _distribution(dices),
        "one_to_many": int(sum(s.get("one_to_many", 0) for s in scores)),
        "many_to_one": int(sum(s.get("many_to_one", 0) for s in scores)),
    }


def bootstrap_macro_pq(scores: Sequence[dict], iterations: int = 1000, seed: int = 2026) -> dict:
    """Bootstrap a record-level confidence interval for macro PQ."""
    if not scores:
        return {"iterations": 0, "mean": 0.0, "ci95": [0.0, 0.0]}
    vals = np.asarray([s["pq"] for s in scores], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(int(iterations), dtype=np.float64)
    for i in range(int(iterations)):
        draws[i] = rng.choice(vals, size=len(vals), replace=True).mean()
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "iterations": int(iterations),
        "mean": float(vals.mean()),
        "ci95": [float(lo), float(hi)],
    }
