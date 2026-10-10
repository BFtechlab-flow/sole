"""Evaluation of cached OOF predictions with one or many post-processing settings.

Used by ``scripts/evaluate_oof.py`` (one setting, full diagnostics) and
``scripts/optimize_postprocess.py`` (many settings, compact counts). Each
image is loaded once and reconstructed with every setting; images are
processed in parallel worker processes.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Sequence

import numpy as np

from .evaluation import group_records_by_image, label_map_to_rles, score_image, subsample_images
from .oof import RECONSTRUCTION_KEYS, cache_path, load_prediction
from .reconstruct import reconstruct_label_map

COUNT_FIELDS = ("iou_sum", "tp", "fp", "fn")


def _work(job):
    fold, file_name, recs, path, param_sets, threshold, rel, detailed = job
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing OOF cache {path}; run scripts/cache_oof.py first")
    pred = load_prediction(path, keys=RECONSTRUCTION_KEYS)
    results = []
    for params in param_sets:
        labels, _ = reconstruct_label_map(
            pred["region"], pred["centerline"], pred["boundary"], pred["distance"],
            orientation=pred.get("orientation"), width=pred.get("width"),
            instance_embedding=pred.get("instance_embedding"), **params,
        )
        rles = label_map_to_rles(labels)
        scores = score_image(recs, rles, threshold, rel)
        for s in scores:
            s["fold"] = int(fold)
            s["file_name"] = file_name
        if detailed:
            counts = [r["counts"].decode("ascii") for r in rles]  # for the official-format export
            for s in scores:
                s["pred_counts"] = counts
            results.append(scores)
        else:
            results.append([(s["fold"], s["stem"], s["scored"], s["iou_sum"], s["tp"], s["fp"], s["fn"]) for s in scores])
    return results


def build_jobs(records, folds, cache_dir, param_sets, metric_cfg=None, max_images_per_fold=None,
               seed=2026, detailed=False, folds_subset: Sequence[int] | None = None):
    metric_cfg = metric_cfg or {}
    threshold = float(metric_cfg.get("iou_match_threshold", 0.5))
    rel = float(metric_cfg.get("relation_iou_threshold", 0.10))
    jobs = []
    fold_ids = sorted(set(int(f) for f in folds)) if folds_subset is None else list(folds_subset)
    for fold in fold_ids:
        idx = np.where(np.asarray(folds) == fold)[0]
        by_file = group_records_by_image(records, idx)
        for file_name in subsample_images(list(by_file), max_images_per_fold, seed=seed + fold):
            recs = [records[i] for i in by_file[file_name]]
            jobs.append((fold, file_name, recs, str(cache_path(cache_dir, fold, file_name)),
                         list(param_sets), threshold, rel, detailed))
    return jobs


def run_jobs(jobs, workers=1, progress=None):
    """Returns, for every parameter set, the concatenated per-entry results."""
    n_sets = len(jobs[0][4]) if jobs else 0
    per_set = [[] for _ in range(n_sets)]
    done = 0

    def consume(res):
        nonlocal done
        for k in range(n_sets):
            per_set[k].extend(res[k])
        done += 1
        if progress and (done % 25 == 0 or done == len(jobs)):
            progress(done, len(jobs))

    if workers <= 1:
        for job in jobs:
            consume(_work(job))
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            for res in ex.map(_work, jobs, chunksize=1):
                consume(res)
    return per_set


def counts_array(rows):
    """Compact rows -> (folds, stems, scored, counts[n, 4])."""
    folds = np.array([r[0] for r in rows], dtype=np.int64)
    stems = [r[1] for r in rows]
    scored = np.array([r[2] for r in rows], dtype=bool)
    counts = np.array([r[3:7] for r in rows], dtype=np.float64).reshape(-1, 4)
    return folds, stems, scored, counts


def pq_from_counts(counts: np.ndarray) -> float:
    s = counts.sum(axis=0)
    den = s[1] + 0.5 * s[2] + 0.5 * s[3]
    return float(s[0] / den) if den > 0 else 0.0


def default_workers():
    n = os.cpu_count() or 1
    return max(1, min(n, 8))
