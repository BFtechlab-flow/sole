import numpy as np
import pandas as pd

from src.rle import encode_counts
from src.submission import validate_submission_frame


def test_submission_validator_accepts_disjoint_instances():
    a = np.zeros((8, 8), np.uint8)
    b = np.zeros((8, 8), np.uint8)
    a[1:3, 1:3] = 1
    b[5:7, 5:7] = 1
    frame = pd.DataFrame([
        {"filament_id": "20260101120000Bh_1", "segmentation_rle": encode_counts(a)},
        {"filament_id": "20260101120000Bh_2", "segmentation_rle": encode_counts(b)},
    ])
    report = validate_submission_frame(frame, {"20260101120000Bh": (8, 8)}, require_nonoverlap=True)
    assert report["ok"]
    assert report["overlap_pixels"] == 0


def test_submission_validator_rejects_overlap():
    a = np.zeros((8, 8), np.uint8)
    b = np.zeros((8, 8), np.uint8)
    a[1:4, 1:4] = 1
    b[3:6, 3:6] = 1
    frame = pd.DataFrame([
        {"filament_id": "20260101120000Bh_1", "segmentation_rle": encode_counts(a)},
        {"filament_id": "20260101120000Bh_2", "segmentation_rle": encode_counts(b)},
    ])
    report = validate_submission_frame(frame, {"20260101120000Bh": (8, 8)}, require_nonoverlap=True)
    assert not report["ok"]
    assert report["overlap_pixels"] > 0


def test_submission_validator_rejects_unknown_image():
    a = np.zeros((8, 8), np.uint8)
    a[1:3, 1:3] = 1
    frame = pd.DataFrame([{"filament_id": "unknown_1", "segmentation_rle": encode_counts(a)}])
    report = validate_submission_frame(frame, {"20260101120000Bh": (8, 8)})
    assert not report["ok"]


def test_submission_validator_rejects_wrong_columns_without_crashing():
    frame = pd.DataFrame([{"wrong": "x"}])
    report = validate_submission_frame(frame, {"20260101120000Bh": (8, 8)})
    assert not report["ok"]
    assert report["errors"]
