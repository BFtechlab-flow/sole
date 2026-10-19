"""Learned instance rescoring: exact separability of the official micro PQ, threshold
selection, feature extraction and the inference hook."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import label_map_to_rles, score_image
from src.pq_official import micro_pq
from src.rescore import (FEATURE_NAMES, apply_rescoring, best_threshold, fit_model, instance_contributions,
                         instance_features, pq_of_selection, rescoring_settings, target)

H = W = 128


def _rect(y0, x0, y1, x1):
    return {"segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]]}


def _scene(seed):
    """Two to three annotators of one image (shared filaments + private ones) and a label map."""
    rng = np.random.default_rng(seed)
    boxes = []
    for _ in range(6):
        y, x = rng.integers(2, H - 30, 2)
        boxes.append((int(y), int(x), int(y + rng.integers(4, 12)), int(x + rng.integers(10, 26))))
    records = []
    for k in range(int(rng.integers(2, 4))):
        mine = [b for b in boxes if rng.random() < 0.7]
        records.append({"height": H, "width": W, "image_id": f"01040{k}-20150125172714Mh",
                        "file_name": "20150125172714Mh.jpg", "annotations": [_rect(*b) for b in mine]})
    labels = np.zeros((H, W), np.int32)
    n = 0
    for b in boxes + [(int(v[0]), int(v[1]), int(v[0]) + 6, int(v[1]) + 9) for v in rng.integers(2, H - 12, (3, 2))]:
        if rng.random() < 0.8:
            y0, x0, y1, x1 = b
            dy, dx = rng.integers(-2, 3, 2)
            region = labels[max(y0 + dy, 0):y1 + dy, max(x0 + dx, 0):x1 + dx]
            n += 1
            region[region == 0] = n  # disjoint, as the reconstruction guarantees
    # drop labels that ended up empty and relabel consecutively
    present = np.unique(labels[labels > 0])
    remap = np.zeros(n + 1, np.int32)
    remap[present] = np.arange(1, len(present) + 1)
    return records, remap[labels]


def _official(records, labels, keep):
    lab = np.where(np.isin(labels, np.flatnonzero(keep) + 1), labels, 0)
    present = np.unique(lab[lab > 0])
    remap = np.zeros(int(labels.max()) + 1, np.int32)
    remap[present] = np.arange(1, len(present) + 1)
    return score_image(records, label_map_to_rles(remap[lab]))


@pytest.mark.parametrize("seed", range(6))
def test_pq_is_separable_per_instance(seed):
    scenes = [_scene(seed * 10 + i) for i in range(3)]
    parts = []
    for records, labels in scenes:
        a, d, g = instance_contributions(score_image(records, label_map_to_rles(labels)), int(labels.max()))
        parts.append((records, labels, a, d, g))
    a_all = np.concatenate([p[2] for p in parts])
    d_all = np.concatenate([p[3] for p in parts])
    g_all = sum(p[4] for p in parts)
    rng = np.random.default_rng(seed)
    for _ in range(5):
        keep = rng.random(len(a_all)) < 0.6
        scores, start = [], 0
        for records, labels, a, _, _ in parts:
            scores += _official(records, labels, keep[start:start + len(a)])
            start += len(a)
        assert pq_of_selection(a_all, d_all, g_all, keep) == pytest.approx(micro_pq(scores), abs=1e-12)


def test_best_threshold_is_the_exhaustive_optimum():
    rng = np.random.default_rng(3)
    for _ in range(20):
        n = int(rng.integers(1, 9))
        a = rng.random(n) * (rng.random(n) < 0.6)
        d = rng.integers(1, 4, n) * 0.5
        g = int(rng.integers(n, 3 * n + 1))
        pred = np.round(rng.random(n), 1)  # ties included
        thr, pq = best_threshold(pred, a, d, g)
        assert pq_of_selection(a, d, g, pred >= thr) == pytest.approx(pq)
        best = max(pq_of_selection(a, d, g, pred >= t) for t in list(pred) + [np.inf])
        assert pq == pytest.approx(best)


def test_target_and_features():
    records, labels = _scene(1)
    rng = np.random.default_rng(0)
    pred = {k: rng.random((H, W)).astype(np.float32) for k in ("region", "centerline", "boundary", "distance")}
    x = instance_features(labels, pred)
    assert x.shape == (int(labels.max()), len(FEATURE_NAMES)) and np.isfinite(x).all()
    assert instance_features(np.zeros((H, W), np.int32), pred).shape == (0, len(FEATURE_NAMES))
    a, d, _ = instance_contributions(score_image(records, label_map_to_rles(labels)), int(labels.max()))
    y = target(a, d)
    assert ((0 <= y) & (y <= 1)).all()


def test_apply_rescoring_keeps_the_instances_above_the_threshold(tmp_path):
    import joblib

    from src.rescore import FEATURES_VERSION

    records, labels = _scene(2)
    pred = {k: np.full((H, W), 0.5, np.float32) for k in ("region", "centerline", "boundary", "distance")}
    x = instance_features(labels, pred)
    y = x[:, FEATURE_NAMES.index("area")] / x[:, FEATURE_NAMES.index("area")].max()  # bigger = better
    model = fit_model(np.repeat(x, 30, axis=0), np.repeat(y, 30))
    path = tmp_path / "r.joblib"
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES, "features_version": FEATURES_VERSION}, path)
    predicted = model.predict(x.astype(np.float64))
    thr = float(np.median(predicted))
    out, scores = apply_rescoring(labels, pred, path, thr)
    kept = np.flatnonzero(predicted >= thr) + 1
    assert int(out.max()) == len(kept) == len(scores)
    for new, old in enumerate(kept, 1):  # same pixels, consecutive labels, order preserved
        assert np.array_equal(out == new, labels == old)

    cfg = {"rescoring": {"enabled": True, "model": str(path), "threshold": thr}}
    assert rescoring_settings(cfg) == (path, thr)
    assert rescoring_settings({"rescoring": {"enabled": False, "model": str(path), "threshold": thr}}) is None
    assert rescoring_settings({}) is None


def test_fit_is_deterministic():
    rng = np.random.default_rng(0)
    x, y = rng.random((200, len(FEATURE_NAMES))), rng.random(200)
    assert np.array_equal(fit_model(x, y).predict(x), fit_model(x, y).predict(x))


def test_contributions_ignore_unscored_entries():
    records, labels = _scene(4)
    empty = dict(records[0], image_id="010409-20150125172714Mh", annotations=[])
    s = score_image(records + [empty], label_map_to_rles(labels))
    a1, d1, g1 = instance_contributions(s, int(labels.max()))
    a0, d0, g0 = instance_contributions(score_image(records, label_map_to_rles(labels)), int(labels.max()))
    assert np.array_equal(a0, a1) and np.array_equal(d0, d1) and g0 == g1
