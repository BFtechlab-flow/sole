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


def test_aggregate_reports_macro_and_micro():
    a = pq_score([mask([(1, 1)])], [mask([(1, 1)])])
    b = pq_score([mask([(2, 2)])], [])
    agg = aggregate_scores([a, b])
    assert agg["macro_pq"] == pytest.approx(0.5)
    assert agg["tp"] == 1 and agg["fn"] == 1
    assert agg["micro_pq"] == pytest.approx(1.0 / 1.5)


def test_non_unique_panoptic_matches_raise():
    gt = [mask([(1, 1), (1, 2), (1, 3)])]
    pred = [gt[0].copy(), gt[0].copy()]
    with pytest.raises(ValueError):
        pq_score(gt, pred)
