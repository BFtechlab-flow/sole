import numpy as np
import torch

from src.losses import discriminative_instance_loss
from src.reconstruct import bridge_skeleton_fragments
from src.topology import h0_persistence_loss, h0_persistence_pairs


def test_h0_persistence_detects_two_long_lived_components():
    field = np.zeros((16, 16), np.float32)
    field[3:6, 3:6] = 0.95
    field[10:13, 10:13] = 0.90
    pairs = h0_persistence_pairs(field)
    persistent = [p for p in pairs if p["birth"] - p["death"] > 0.5]
    assert len(persistent) == 2


def test_h0_loss_penalizes_fragmentation_and_has_gradient():
    target = torch.zeros(1, 1, 32, 32)
    target[:, :, 14:18, 4:28] = 1.0

    good = torch.full_like(target, -8.0, requires_grad=True)
    with torch.no_grad():
        good[:, :, 14:18, 4:28] = 8.0
    broken = good.detach().clone().requires_grad_(True)
    with torch.no_grad():
        broken[:, :, 14:18, 15:18] = -8.0

    good_loss = h0_persistence_loss(good, target, max_size=32, max_features=32)
    broken_loss = h0_persistence_loss(broken, target, max_size=32, max_features=32)
    assert broken_loss > good_loss
    broken_loss.backward()
    assert broken.grad is not None
    assert torch.isfinite(broken.grad).all()


def test_discriminative_embedding_prefers_separated_instance_centers():
    ids = torch.zeros(1, 1, 8, 8, dtype=torch.long)
    ids[:, :, 1:4, 1:4] = 1
    ids[:, :, 4:7, 4:7] = 2

    good = torch.zeros(1, 2, 8, 8)
    good[:, 0, 1:4, 1:4] = -2.0
    good[:, 0, 4:7, 4:7] = 2.0
    bad = torch.zeros_like(good)

    good_loss = discriminative_instance_loss(good, ids)
    bad_loss = discriminative_instance_loss(bad, ids)
    assert good_loss < bad_loss


def test_geodesic_bridge_routes_around_high_boundary_cost():
    sk = np.zeros((64, 64), dtype=bool)
    sk[32, 8:24] = True
    sk[32, 41:56] = True

    region = np.ones((64, 64), dtype=np.float32)
    boundary = np.zeros_like(region)
    boundary[24:41, 30:35] = 1.0
    orientation = np.zeros((2, 64, 64), dtype=np.float32)
    orientation[0] = 1.0

    _, bridge_mask, accepted = bridge_skeleton_fragments(
        sk,
        region_prob=region,
        boundary_prob=boundary,
        orientation=orientation,
        max_distance=24,
        min_alignment=0.0,
        min_region=0.0,
        max_boundary=0.35,
        geodesic=True,
        geodesic_padding=20,
        geodesic_boundary_weight=20.0,
        max_geodesic_ratio=2.5,
    )
    assert accepted
    assert bridge_mask.any()
    assert not np.any(bridge_mask & (boundary > 0.5))


def test_embedding_affinity_rejects_wrong_fragment_merge():
    sk = np.zeros((64, 64), dtype=bool)
    sk[32, 8:26] = True
    sk[32, 31:52] = True
    region = np.ones((64, 64), dtype=np.float32)
    boundary = np.zeros_like(region)
    orientation = np.zeros((2, 64, 64), dtype=np.float32)
    orientation[0] = 1.0
    embedding = np.zeros((2, 64, 64), dtype=np.float32)
    embedding[0, 30:35, 7:27] = 1.0
    embedding[0, 30:35, 30:53] = -1.0

    _, bridge_mask, accepted = bridge_skeleton_fragments(
        sk,
        region_prob=region,
        boundary_prob=boundary,
        orientation=orientation,
        instance_embedding=embedding,
        max_distance=10,
        min_alignment=0.8,
        min_region=0.0,
        min_embedding_similarity=0.0,
    )
    assert not bridge_mask.any()
    assert accepted == []
