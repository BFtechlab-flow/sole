"""Learned instance rescoring: drop the predicted filaments that lower the official PQ.

Why it is exact to decide instance by instance
----------------------------------------------
The official score is a micro PQ ``S / D`` summed over annotator entries, with
``S = sum of matched IoUs`` and ``D = TP + FP/2 + FN/2``. Predictions are
disjoint, so a ground-truth filament can be hit (IoU > 0.5) by at most one of
them. Removing a predicted instance ``j`` therefore changes nothing else: in
every entry ``e`` of its image it either was a false positive (``D`` loses
1/2) or hit ``h_je`` ground-truth filaments, which become false negatives
(``S`` loses their IoUs, ``D`` loses ``h_je / 2``). With

    a_j = sum_e (IoUs of the hits of j in entry e)
    d_j = sum_e (1/2 if j hits nothing in e else h_je / 2)

and ``G`` the number of ground-truth filaments, keeping the set ``K`` gives

    PQ(K) = sum_{j in K} a_j / (G / 2 + sum_{j in K} d_j)

exactly (checked against the official scorer in ``tests/test_rescore.py``).
An instance is worth keeping iff ``a_j / d_j > PQ``; the regression target is
``y_j = a_j / (2 d_j)`` (the mean matched IoU over the entries of its image),
and instances with a predicted ``y`` below a threshold are dropped.

The scorer is a gradient-boosted regressor on features of each instance that
only use the prediction maps (never the image of another observation, never a
label of the test set). It is cross-fitted on the OOF predictions, and its
threshold is chosen by an inner cross-validation, so the nested estimate of
``scripts/train_rescorer.py`` never scores a fold with a model or threshold
that saw it.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, find_objects
from skimage.morphology import skeletonize

FEATURES_VERSION = 2
FEATURE_NAMES = (
    "area", "log_area",
    "region_mean", "region_max", "region_p10", "region_frac_gt_0p7",
    "center_mean", "center_max",
    "boundary_mean",
    "distance_mean", "distance_max",
    "width_mean",
    "skeleton_length", "elongation", "extent_major", "extent_minor", "fill_ratio",
    "radial_position",
    "ring_region_mean", "ring_contrast", "ring_touching_other",
    "image_instances", "image_score_rank", "image_foreground_fraction", "nearest_instance_distance",
)
RING = 3  # pixels
# Features of the H-alpha test image itself (filaments are dark, elongated absorption features).
# Only the image being segmented is used, as the competition requires.
IMAGE_FEATURE_NAMES = (
    "img_mean", "img_p10", "img_std", "img_ring_mean", "img_contrast", "img_background_contrast",
    "img_ridge_mean",
)
FEATURE_SETS = {"maps": FEATURE_NAMES, "maps+image": FEATURE_NAMES + IMAGE_FEATURE_NAMES}
IMG_RING, IMG_BACKGROUND = 4, 15  # pixels


def instance_features(labels: np.ndarray, pred: dict) -> np.ndarray:
    """``(n_instances, len(FEATURE_NAMES))`` float32 features, in label order (1..N)."""
    n = int(labels.max())
    out = np.zeros((n, len(FEATURE_NAMES)), np.float32)
    if n == 0:
        return out
    h, w = labels.shape
    region = np.asarray(pred["region"], np.float32)
    center = np.asarray(pred["centerline"], np.float32)
    boundary = np.asarray(pred["boundary"], np.float32)
    distance = np.asarray(pred["distance"], np.float32)
    width = pred.get("width")
    width = np.asarray(width, np.float32) if width is not None and np.ndim(width) == 2 else None
    centroids = np.zeros((n, 2), np.float64)
    pad = RING + 1
    for i, sl in enumerate(find_objects(labels)):
        if sl is None:
            continue
        y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, h)
        x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, w)
        lab = labels[y0:y1, x0:x1]
        m = lab == i + 1
        area = float(m.sum())
        r = region[y0:y1, x0:x1]
        r_in = r[m]
        skel = float(skeletonize(m).sum())
        ring = binary_dilation(m, iterations=RING) & ~m
        ring_r = float(r[ring].mean()) if ring.any() else 0.0
        touching = float(((lab[ring] > 0) & (lab[ring] != i + 1)).mean()) if ring.any() else 0.0
        ys, xs = np.nonzero(m)
        cy, cx = ys.mean() + y0, xs.mean() + x0
        centroids[i] = cy, cx
        ext = sorted((sl[0].stop - sl[0].start, sl[1].stop - sl[1].start))
        out[i, :21] = (
            area, np.log1p(area),
            r_in.mean(), r_in.max(), np.percentile(r_in, 10), (r_in > 0.7).mean(),
            center[y0:y1, x0:x1][m].mean(), center[y0:y1, x0:x1][m].max(),
            boundary[y0:y1, x0:x1][m].mean(),
            distance[y0:y1, x0:x1][m].mean(), distance[y0:y1, x0:x1][m].max(),
            width[y0:y1, x0:x1][m].mean() if width is not None else 0.0,
            skel, skel * skel / max(area, 1.0), ext[1], ext[0], area / max(ext[0] * ext[1], 1),
            np.hypot(cy - (h - 1) / 2, cx - (w - 1) / 2) / (min(h, w) / 2),
            ring_r, float(r_in.mean()) - ring_r, touching,
        )
    scores = out[:, 2]
    rank = np.argsort(np.argsort(-scores, kind="stable"), kind="stable")
    out[:, 21] = n
    out[:, 22] = rank / max(n - 1, 1)
    out[:, 23] = float((region > 0.45).mean())
    if n > 1:
        dist = np.hypot(*(centroids[:, None, :] - centroids[None, :, :]).transpose(2, 0, 1))
        np.fill_diagonal(dist, np.inf)
        out[:, 24] = dist.min(axis=1) / min(h, w)
    else:
        out[:, 24] = 1.0
    return out


def normalised_image(gray: np.ndarray):
    """Robustly normalised intensity (1st-99th percentile -> 0-1) and its ridge response (DoG)."""
    x = np.asarray(gray, np.float32)
    lo, hi = np.percentile(x, [1, 99])
    raw = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)
    ridge = cv2.GaussianBlur(raw, (0, 0), 5.0) - cv2.GaussianBlur(raw, (0, 0), 1.2)  # dark ridges -> positive
    return raw, ridge


def image_features(labels: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """``(n_instances, len(IMAGE_FEATURE_NAMES))`` features of the grayscale image, in label order."""
    n = int(labels.max())
    out = np.zeros((n, len(IMAGE_FEATURE_NAMES)), np.float32)
    if n == 0:
        return out
    if gray.shape != labels.shape:
        raise ValueError(f"image {gray.shape} and labels {labels.shape} differ")
    raw, ridge = normalised_image(gray)
    h, w = labels.shape
    pad = IMG_BACKGROUND + 1
    k_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * IMG_RING + 1, 2 * IMG_RING + 1))
    k_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * IMG_BACKGROUND + 1, 2 * IMG_BACKGROUND + 1))
    for i, sl in enumerate(find_objects(labels)):
        if sl is None:
            continue
        y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, h)
        x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, w)
        m = (labels[y0:y1, x0:x1] == i + 1).astype(np.uint8)
        near = cv2.dilate(m, k_ring).astype(bool)
        far = cv2.dilate(m, k_bg).astype(bool)
        inside = m.astype(bool)
        ring = near & ~inside
        background = far & ~near
        r = raw[y0:y1, x0:x1]
        v = r[inside]
        ring_mean = float(r[ring].mean()) if ring.any() else float(v.mean())
        bg_mean = float(r[background].mean()) if background.any() else ring_mean
        out[i] = (v.mean(), np.percentile(v, 10), v.std(), ring_mean, ring_mean - v.mean(), bg_mean - v.mean(),
                  ridge[y0:y1, x0:x1][inside].mean())
    return out


def features_for(labels: np.ndarray, pred: dict, feature_set: str = "maps", image=None) -> np.ndarray:
    if feature_set == "maps":
        return instance_features(labels, pred)
    if feature_set == "maps+image":
        if image is None:
            raise ValueError("this rescorer needs the grayscale image (feature set 'maps+image')")
        return np.hstack([instance_features(labels, pred), image_features(labels, image)])
    raise ValueError(f"unknown feature set {feature_set!r}")


def instance_contributions(entry_scores, n_instances: int):
    """``a`` (sum of matched IoUs) and ``d`` (denominator share) of every instance, plus ``G``.

    ``entry_scores`` are the per-entry results of ``src.evaluation.score_image``
    for the instances of one image (prediction ``j`` = label ``j + 1``).
    Unscored entries (no ground truth) are ignored, as by the official scorer.
    """
    a = np.zeros(n_instances, np.float64)
    d = np.zeros(n_instances, np.float64)
    g = 0
    for s in entry_scores:
        if not s["scored"]:
            continue
        g += int(s["gt_count"])
        hits = np.zeros(n_instances, np.int64)
        for _, j, iou in s["pairs"]:
            a[j] += iou
            hits[j] += 1
        d += np.where(hits == 0, 0.5, 0.5 * hits)
    return a, d, g


def pq_of_selection(a, d, g_total, keep) -> float:
    keep = np.asarray(keep, bool)
    den = 0.5 * g_total + float(np.asarray(d)[keep].sum())
    return float(np.asarray(a)[keep].sum() / den) if den > 0 else 0.0


def best_threshold(pred_y, a, d, g_total):
    """Threshold on the predicted ``y`` maximising the PQ of the kept instances.

    Returns ``(threshold, pq)``; instances with ``pred_y >= threshold`` are kept.
    """
    pred_y, a, d = map(np.asarray, (pred_y, a, d))
    if pred_y.size == 0:
        return 0.0, 0.0
    order = np.argsort(-pred_y, kind="stable")
    ys = pred_y[order]
    cum_a = np.concatenate([[0.0], np.cumsum(a[order])])
    cum_d = np.concatenate([[0.0], np.cumsum(d[order])])
    pq = cum_a / np.maximum(0.5 * g_total + cum_d, 1e-12)
    # only cut between distinct predicted values (all tied instances are kept or dropped together)
    valid = np.ones(len(pq), bool)
    valid[1:-1] = ys[:-1] != ys[1:]
    k = int(np.argmax(np.where(valid, pq, -1.0)))
    if k == 0:
        thr = float(ys[0]) + 1e-6  # keep nothing
    elif k == len(ys):
        thr = float(ys[-1]) - 1e-6 if ys[-1] > 0 else 0.0  # keep everything
    else:
        thr = float((ys[k - 1] + ys[k]) / 2)
    return thr, float(pq[k])


def make_model(seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingRegressor

    # no early stopping: its random validation split would make the fit depend on more than the seed
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=40, l2_regularization=1.0,
                                         early_stopping=False, random_state=seed)


def fit_model(x, y, seed: int = 0):
    model = make_model(seed)
    model.fit(np.asarray(x, np.float64), np.asarray(y, np.float64))
    return model


def target(a, d):
    return np.where(np.asarray(d) > 0, np.asarray(a) / (2.0 * np.maximum(d, 1e-12)), 0.0)


# ----------------------------------------------------------------------------- inference
_LOADED: dict = {}


def load_rescorer(path):
    import joblib

    path = Path(path)
    key = str(path.resolve())
    if key not in _LOADED:
        bundle = joblib.load(path)
        feature_set = bundle.setdefault("feature_set", "maps")
        if (bundle.get("features_version") != FEATURES_VERSION or feature_set not in FEATURE_SETS
                or tuple(bundle["feature_names"]) != FEATURE_SETS[feature_set]):
            raise ValueError(f"{path}: rescorer built for other features; rerun scripts/train_rescorer.py")
        _LOADED[key] = bundle
    return _LOADED[key]


def candidate_overrides(cfg) -> dict:
    """Reconstruction settings replaced when the rescoring is enabled (e.g. a permissive candidate set)."""
    sec = (cfg or {}).get("rescoring") or {}
    return dict(sec.get("candidate_overrides") or {}) if sec.get("enabled") and sec.get("model") else {}


def rescoring_settings(cfg):
    """``(model_path, threshold)`` of the config's ``rescoring`` section, or ``None`` when disabled.

    A relative model path is read from the working directory, else from the repository root.
    """
    sec = (cfg or {}).get("rescoring") or {}
    if not sec.get("enabled") or not sec.get("model"):
        return None
    path = Path(sec["model"])
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).resolve().parents[1] / path
    return path, float(sec["threshold"])


def apply_rescoring(labels: np.ndarray, pred: dict, model_path, threshold: float, image=None):
    """Drop the instances whose predicted contribution is below ``threshold``; relabel 1..K.

    ``image`` (the grayscale image being segmented) is required by 'maps+image' rescorers.
    """
    n = int(labels.max())
    if n == 0:
        return labels, np.zeros(0, np.float32)
    bundle = load_rescorer(model_path)
    y = bundle["model"].predict(features_for(labels, pred, bundle["feature_set"], image).astype(np.float64))
    keep = y >= float(threshold)
    remap = np.zeros(n + 1, np.int32)
    remap[1:][keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return remap[labels], y[keep].astype(np.float32)
