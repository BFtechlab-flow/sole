import numpy as np
import torch

from src.data import curvature_from_orientation, skeleton_keypoints
from src.losses import (
    centerline_ce_loss,
    masked_smooth_l1,
    uncertainty_region_loss,
)
from src.model import FILANet
from src.reconstruct import bridge_skeleton_fragments


def test_advanced_model_heads_cpu():
    advanced = {
        "enabled": True,
        "limb_gate": True,
        "multiscale_context": True,
        "frequency_branch": True,
        "width_head": True,
        "curvature_head": True,
        "endpoint_head": True,
        "junction_head": True,
        "uncertainty_head": True,
    }
    model = FILANet(
        encoder="convnext_tiny.fb_in1k",
        pretrained=False,
        fpn_channels=32,
        in_chans=4,
        advanced=advanced,
    ).eval()
    x = torch.randn(1, 4, 96, 96)
    with torch.no_grad():
        out = model(x)
    for key in ("width", "curvature", "endpoint", "junction", "uncertainty"):
        assert key in out
        assert out[key].shape == (1, 1, 96, 96)
        assert torch.isfinite(out[key]).all()


def test_endpoint_and_junction_targets():
    sk = np.zeros((15, 15), dtype=np.uint8)
    sk[3:12, 7] = 1
    sk[7, 4:11] = 1
    endpoint, junction = skeleton_keypoints(sk, dilation_radius=0)
    assert int(endpoint.sum()) == 4
    assert junction[7, 7] == 1


def test_curvature_straight_line_is_small():
    sk = np.zeros((21, 21), dtype=np.uint8)
    sk[10, 3:18] = 1
    orientation = np.zeros((2, 21, 21), np.float32)
    orientation[0, 10, 3:18] = 1.0
    valid = sk.astype(np.float32)
    curv, curv_valid = curvature_from_orientation(orientation, valid)
    assert curv_valid.sum() > 0
    assert float(curv[curv_valid > 0].max()) < 1e-3


def test_advanced_losses_are_finite():
    logits = torch.zeros((1, 1, 32, 32), requires_grad=True)
    target = torch.zeros_like(logits)
    target[:, :, 15:17, 5:27] = 1
    valid = torch.ones_like(logits)
    l1 = centerline_ce_loss(logits, target)
    l2 = masked_smooth_l1(logits, target, valid)
    l3 = uncertainty_region_loss(logits, target, torch.zeros_like(logits))
    total = l1 + l2 + l3
    total.backward()
    assert torch.isfinite(total)
    assert logits.grad is not None


def test_width_mismatch_rejects_bridge():
    sk = np.zeros((64, 64), dtype=bool)
    sk[32, 10:24] = True
    sk[32, 30:44] = True
    region = np.ones((64, 64), np.float32)
    boundary = np.zeros((64, 64), np.float32)
    orientation = np.zeros((2, 64, 64), np.float32)
    orientation[0] = 1.0
    width = np.zeros((64, 64), np.float32)
    width[30:35, 8:25] = 0.10
    width[30:35, 29:46] = 0.80

    bridged, bridge_mask, accepted = bridge_skeleton_fragments(
        sk,
        region_prob=region,
        boundary_prob=boundary,
        orientation=orientation,
        width=width,
        max_distance=12,
        min_alignment=0.8,
        max_width_ratio=2.0,
    )
    assert not bridge_mask.any()
    assert accepted == []
    assert np.array_equal(bridged, sk)
