import numpy as np
from skimage.measure import label

from src.reconstruct import bridge_skeleton_fragments


def test_graph_bridge_connects_aligned_fragments():
    sk = np.zeros((64, 64), dtype=bool)
    sk[32, 8:26] = True
    sk[32, 31:52] = True

    region = np.zeros((64, 64), dtype=np.float32)
    region[30:35, 7:53] = 0.9
    boundary = np.zeros_like(region)
    orientation = np.zeros((2, 64, 64), dtype=np.float32)
    orientation[0] = 1.0

    bridged, bridge_mask, accepted = bridge_skeleton_fragments(
        sk,
        region_prob=region,
        boundary_prob=boundary,
        orientation=orientation,
        max_distance=10,
        min_alignment=0.9,
        min_region=0.2,
    )
    assert bridge_mask.any()
    assert len(accepted) == 1
    assert label(bridged, connectivity=2).max() == 1


def test_graph_bridge_rejects_misaligned_fragments():
    sk = np.zeros((64, 64), dtype=bool)
    sk[20, 20:35] = True
    sk[28, 20:35] = True

    region = np.ones((64, 64), dtype=np.float32)
    boundary = np.zeros_like(region)
    orientation = np.zeros((2, 64, 64), dtype=np.float32)
    orientation[0] = 1.0

    _, bridge_mask, accepted = bridge_skeleton_fragments(
        sk,
        region_prob=region,
        boundary_prob=boundary,
        orientation=orientation,
        max_distance=10,
        min_alignment=0.9,
        min_region=0.2,
    )
    assert not bridge_mask.any()
    assert accepted == []
