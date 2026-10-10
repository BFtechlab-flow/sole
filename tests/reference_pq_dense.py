"""Independent, deliberately naive implementation of the competition PQ.

Written from the evaluation rules published by the organizers (Overview >
Leaderboard Ranking, and the algorithm described by their public
Self_Evaluation_Notebook), NOT copied from their code. It is the test oracle
for ``src.pq_official``, which is fast (run-length encodings) but more
intricate. Everything here is dense and step by step:

1. ground-truth rows are grouped by annotator-image entry (text of
   ``filament_id`` before the first ``_``, e.g. ``010401-20160920230134Lh``);
2. prediction rows are grouped by image id (text before the first ``_``);
3. each entry is compared with every prediction of its image (the entry's
   image id is the text after the first ``-``): IoU of every (GT, prediction)
   pair, computed from dense float32 masks; IoU is 0 when the union is empty;
4. a pair is a hit when IoU > 0.5; every hit is a true positive worth its IoU;
   a prediction without hit is a false positive, a GT filament without hit a
   false negative; an entry without predictions makes all its GT filaments
   false negatives;
5. TP / FP / FN and the IoU sum are accumulated over all entries and
   ``PQ = sum IoU(TP) / (TP + FP/2 + FN/2)`` is computed once (0 if the
   denominator is 0). Predictions for images absent from the ground truth
   never enter the computation.

To check our implementation against the organizers' own code, use
``tools/verify_against_official_notebook.py`` with the notebook downloaded
from Kaggle (it is not redistributed in this repository).
"""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils


def _decode_stack(counts_list, height, width):
    masks = np.zeros((len(counts_list), height, width), dtype=np.float32)
    for k, counts in enumerate(counts_list):
        rle = {"size": [height, width], "counts": counts.encode("ascii") if isinstance(counts, str) else counts}
        masks[k] = mask_utils.decode(rle)
    return masks


def dense_iou(gt_masks: np.ndarray, pred_masks: np.ndarray) -> np.ndarray:
    """IoU matrix (rows = GT, columns = predictions) from dense float32 masks."""
    n_gt, n_pred = gt_masks.shape[0], pred_masks.shape[0]
    iou = np.zeros((n_gt, n_pred), dtype=np.float32)
    for i in range(n_gt):
        for j in range(n_pred):
            inter = np.float32((gt_masks[i] * pred_masks[j]).sum(dtype=np.float32))
            union = np.float32(gt_masks[i].sum(dtype=np.float32) + pred_masks[j].sum(dtype=np.float32) - inter)
            iou[i, j] = np.float32(0.0) if union == 0 else np.float32(inter / union)
    return iou


def reference_pq(gt_rows, pred_rows, height=2048, width=2048, threshold=0.5):
    """``gt_rows`` / ``pred_rows``: iterables of ``(filament_id, rle_counts)``."""
    entries: dict[str, list[str]] = {}
    for filament_id, counts in gt_rows:
        entries.setdefault(filament_id.split("_", 1)[0], []).append(counts)
    images: dict[str, list[str]] = {}
    for filament_id, counts in pred_rows:
        images.setdefault(filament_id.split("_", 1)[0], []).append(counts)

    iou_sum, tp, fp, fn = 0.0, 0, 0, 0
    for entry, gt_counts in entries.items():
        image_id = entry.split("-", 1)[1]
        pred_counts = images.get(image_id, [])
        if not pred_counts:
            fn += len(gt_counts)
            continue
        iou = dense_iou(_decode_stack(gt_counts, height, width), _decode_stack(pred_counts, height, width))
        for i in range(iou.shape[0]):
            for j in range(iou.shape[1]):
                if iou[i, j] > threshold:
                    tp += 1
                    iou_sum += float(iou[i, j])
        fp += sum(1 for j in range(iou.shape[1]) if not (iou[:, j] > threshold).any())
        fn += sum(1 for i in range(iou.shape[0]) if not (iou[i, :] > threshold).any())
    den = tp + 0.5 * fp + 0.5 * fn
    return {"pq": iou_sum / den if den > 0 else 0.0, "tp": tp, "fp": fp, "fn": fn, "iou_sum": iou_sum}
