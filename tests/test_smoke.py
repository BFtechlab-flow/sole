import numpy as np
import torch

from src.data import build_input_channels
from src.metric import pq_score
from src.model import FILANet
from src.reconstruct import reconstruct_instances
from src.rle import assert_roundtrip, decode_counts


def test_rle_roundtrip():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, 20:40] = 1
    counts = assert_roundtrip(mask)
    decoded = decode_counts(counts, 64, 64)
    assert np.array_equal(mask, decoded)


def test_pq_perfect_match():
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[5:20, 8:24] = 1
    score = pq_score([mask], [mask], threshold=0.5)
    assert score["pq"] == 1.0
    assert score["tp"] == 1
    assert score["fp"] == 0
    assert score["fn"] == 0


def test_preprocessing_shape_and_range():
    gray = np.tile(np.arange(128, dtype=np.uint8), (128, 1))
    x = build_input_channels(gray)
    assert x.shape == (4, 128, 128)
    assert np.isfinite(x).all()
    assert x.min() >= 0.0
    assert x.max() <= 1.0


def test_reconstruction_synthetic_filament():
    h = w = 128
    region = np.zeros((h, w), dtype=np.float32)
    center = np.zeros_like(region)
    boundary = np.zeros_like(region)
    distance = np.zeros_like(region)
    region[58:70, 20:108] = 0.95
    center[63:65, 22:106] = 0.95
    distance[59:69, 21:107] = 0.9
    masks = reconstruct_instances(
        region,
        center,
        boundary,
        distance,
        region_threshold=0.45,
        center_threshold=0.35,
        min_region_area=32,
        min_instance_area=100,
    )
    assert len(masks) >= 1
    assert max(int(m.sum()) for m in masks) >= 100


def test_model_forward_cpu():
    model = FILANet(
        encoder="convnext_tiny.fb_in1k",
        pretrained=False,
        fpn_channels=32,
        in_chans=4,
    ).eval()
    x = torch.randn(1, 4, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert set(out) == {"region", "centerline", "boundary", "distance"}
    for value in out.values():
        assert value.shape == (1, 1, 128, 128)
        assert torch.isfinite(value).all()
