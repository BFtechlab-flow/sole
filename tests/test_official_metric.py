import numpy as np
import pytest

from src.pq_official import aggregate_scores, pq_score


def mask(coords, shape=(8, 8)):
    out = np.zeros(shape, np.uint8)
    for y, x in coords:
        out[y, x] = 1
    return out


def test_perfect_match_is_one():
    gt = [mask([(1, 1), (1, 2), (2, 1), (2, 2)])]
    score = pq_score(gt, [gt[0].copy()])
    assert score["pq"] == pytest.approx(1.0)
    assert score["tp"] == 1 and score["fp"] == 0 and score["fn"] == 0


def test_match_threshold_is_strictly_greater_than_half():
    gt = [mask([(1, 1), (1, 2)])]
    pred = [mask([(1, 1)])]  # IoU = exactly 0.5
    score = pq_score(gt, pred, threshold=0.5)
    assert score["tp"] == 0
    assert score["fp"] == 1
    assert score["fn"] == 1
    assert score["pq"] == 0.0


def test_empty_prediction_is_penalized_when_gt_exists():
    gt = [mask([(1, 1), (1, 2)])]
    score = pq_score(gt, [])
    assert score["pq"] == 0.0
    assert score["tp"] == 0 and score["fp"] == 0 and score["fn"] == 1


def test_aggregate_official_micro_and_macro_diagnostic():
    a = pq_score([mask([(1, 1)])], [mask([(1, 1)])])
    b = pq_score([mask([(2, 2)])], [])
    agg = aggregate_scores([a, b])
    assert agg["tp"] == 1 and agg["fn"] == 1
    # official score = global sums (micro)
    assert agg["pq"] == pytest.approx(1.0 / 1.5)
    assert agg["micro_pq"] == agg["pq"]
    # mean of per-entry PQ is only a diagnostic
    assert agg["macro_pq_diagnostic"] == pytest.approx(0.5)


def test_overlapping_duplicates_are_counted_like_the_official_notebook():
    # The organizers' code has no one-to-one assignment: both duplicates hit.
    gt = [mask([(1, 1), (1, 2), (1, 3)])]
    pred = [gt[0].copy(), gt[0].copy()]
    score = pq_score(gt, pred)
    assert (score["tp"], score["fp"], score["fn"]) == (2, 0, 0)


def test_relations_use_any_overlap_like_the_official_plot():
    gt = [mask([(1, 1), (1, 2), (1, 3), (1, 4)])]
    pred = [mask([(1, 1), (1, 2)]), mask([(1, 3), (1, 4)])]  # split filament
    score = pq_score(gt, pred)
    assert score["gt_degree"] == [2]
    assert score["one_to_many"] == 1
    assert (score["tp"], score["fp"], score["fn"]) == (0, 2, 1)
