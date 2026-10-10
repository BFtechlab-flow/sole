"""Parity of ``src.pq_official`` with an independent dense implementation.

``tests/reference_pq_dense.py`` re-implements the published competition
algorithm naively (dense float32 masks, explicit loops). Random
multi-annotator scenes (missed, split, merged, shifted, dilated and spurious
filaments, empty prediction sets, duplicated predictions, predictions on
images absent from the ground truth) are scored with both implementations;
PQ and TP/FP/FN must agree. The check against the organizers' own notebook
code is ``tools/verify_against_official_notebook.py`` (the notebook is not
redistributed here).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi

from src.pq_official import aggregate_scores, evaluate_frames, iou_matrix, pq_score
from src.rle import encode_counts

from tests.reference_pq_dense import reference_pq

SHAPE = (96, 112)  # non-square on purpose (catches H/W swaps)


def random_filament(rng, shape=SHAPE):
    h, w = shape
    m = np.zeros(shape, dtype=bool)
    y, x = int(rng.integers(8, h - 8)), int(rng.integers(8, w - 8))
    ang = rng.uniform(0, 2 * np.pi)
    for _ in range(int(rng.integers(8, 30))):
        ang += rng.normal(0, 0.35)
        y = int(np.clip(y + 2 * np.sin(ang), 0, h - 1))
        x = int(np.clip(x + 2 * np.cos(ang), 0, w - 1))
        m[y, x] = True
    return ndi.binary_dilation(m, iterations=int(rng.integers(1, 3)))


def perturb(rng, m):
    op = int(rng.integers(0, 6))
    if op == 0:
        return []
    if op == 1:
        return [np.roll(m, (int(rng.integers(-2, 3)), int(rng.integers(-2, 3))), axis=(0, 1))]
    if op == 2:
        return [ndi.binary_dilation(m, iterations=int(rng.integers(1, 3)))]
    if op == 3:
        e = ndi.binary_erosion(m)
        return [e] if e.any() else [m]
    if op == 4:
        ys, xs = np.nonzero(m)
        if len(ys) < 4:
            return [m]
        idx = xs if np.ptp(xs) >= np.ptp(ys) else ys
        cut = np.median(idx)
        a, b = m.copy(), m.copy()
        a[ys[idx > cut], xs[idx > cut]] = False
        b[ys[idx <= cut], xs[idx <= cut]] = False
        return [p for p in (a, b) if p.any()]
    return [m.copy()]


def disjoint(masks):
    taken = np.zeros(SHAPE, bool)
    out = []
    for m in masks:
        m = m & ~taken
        taken |= m
        if m.any():
            out.append(m)
    return out


def scene(rng, stem, allow_overlap):
    base = [random_filament(rng) for _ in range(int(rng.integers(1, 6)))]
    gt_rows, pred_rows = [], []
    for a in range(int(rng.integers(1, 4))):
        entry = f"0104{a:02d}-{stem}"
        kept = [f if rng.random() < 0.5 else ndi.binary_dilation(f) for f in base if rng.random() > 0.2]
        kept = kept or [base[0]]
        gt_rows += [(f"{entry}_{k}", encode_counts(m)) for k, m in enumerate(kept, 1)]
    preds = [p for f in base for p in perturb(rng, f)]
    preds += [random_filament(rng) for _ in range(int(rng.integers(0, 3)))]
    preds = preds if allow_overlap else disjoint(preds)
    if allow_overlap and preds and rng.random() < 0.5:
        preds.append(preds[0].copy())
    if rng.random() < 0.15:
        preds = []
    pred_rows += [(f"{stem}_{k}", encode_counts(p)) for k, p in enumerate(preds, 1) if p.any()]
    return gt_rows, pred_rows


def reference(gt_df, pred_df):
    return reference_pq(gt_df.itertuples(index=False, name=None), pred_df.itertuples(index=False, name=None),
                        height=SHAPE[0], width=SHAPE[1])


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("allow_overlap", [False, True])
def test_pq_matches_independent_reference(seed, allow_overlap):
    rng = np.random.default_rng(seed)
    gt_rows, pred_rows = [], []
    for i in range(int(rng.integers(3, 8))):
        g, p = scene(rng, f"2015012517{i:04d}Mh", allow_overlap)
        gt_rows += g
        pred_rows += p
    pred_rows.append(("20990101000000Xx_1", encode_counts(random_filament(rng))))
    cols = ["filament_id", "segmentation_rle"]
    gt_df, pred_df = pd.DataFrame(gt_rows, columns=cols), pd.DataFrame(pred_rows, columns=cols)

    ours = evaluate_frames(gt_df, pred_df, shape=SHAPE)
    ref = reference(gt_df, pred_df)
    assert (ours["tp"], ours["fp"], ours["fn"]) == (ref["tp"], ref["fp"], ref["fn"])
    assert ours["pq"] == pytest.approx(ref["pq"], abs=1e-9)


def test_iou_matrix_is_bit_identical_to_dense_float32():
    rng = np.random.default_rng(3)
    gt = [random_filament(rng) for _ in range(6)]
    pred = [random_filament(rng) for _ in range(5)] + [gt[0].copy(), ndi.binary_dilation(gt[1])]
    ours = iou_matrix(gt, pred)
    g = np.stack(gt).reshape(len(gt), -1).astype(np.float32)
    p = np.stack(pred).reshape(len(pred), -1).astype(np.float32)
    inter = g @ p.T
    union = g.sum(1)[:, None] + p.sum(1)[None, :] - inter
    dense = np.where(union == 0, np.float32(0), inter / union).astype(np.float32)
    assert ours.dtype == np.float32
    assert np.array_equal(ours, dense)


def test_hand_computed_cases():
    s = (20, 20)

    def box(y0, y1, x0, x1):
        m = np.zeros(s, np.uint8)
        m[y0:y1, x0:x1] = 1
        return m

    g = box(0, 10, 0, 10)
    assert pq_score([g], [g])["pq"] == pytest.approx(1.0)
    r = pq_score([g], [])
    assert (r["tp"], r["fp"], r["fn"], r["pq"]) == (0, 0, 1, 0.0)
    r = pq_score([g], [box(0, 5, 0, 10)])  # IoU exactly 0.5 -> not a match
    assert (r["tp"], r["fp"], r["fn"]) == (0, 1, 1)
    assert pq_score([g], [box(0, 6, 0, 10)])["pq"] == pytest.approx(0.6)

    # micro aggregation: entry A perfect, entry B missed -> 1 / (1 + 0.5)
    a = pq_score([g], [g])
    b = pq_score([box(12, 20, 12, 20)], [])
    agg = aggregate_scores([a, b])
    assert agg["pq"] == pytest.approx(1 / 1.5)
    assert agg["macro_pq_diagnostic"] == pytest.approx(0.5)

    # entries without GT are not scored by the official metric
    c = pq_score([], [g])
    assert c["scored"] is False
    assert aggregate_scores([a, c])["pq"] == pytest.approx(1.0)

    # official quirk: overlapping duplicates are both counted (no exception)
    r = pq_score([g], [g, g.copy()])
    assert (r["tp"], r["fp"], r["fn"]) == (2, 0, 0)
