import numpy as np

from src.data import apply_geometric_transform, orientation_from_skeleton


def test_orientation_target_horizontal_and_vertical():
    horizontal = np.zeros((21, 21), dtype=bool)
    horizontal[10, 3:18] = True
    ori_h, valid_h = orientation_from_skeleton(horizontal, radius=3)
    mask_h = valid_h > 0
    assert mask_h.any()
    assert np.mean(ori_h[0][mask_h]) > 0.95
    assert abs(float(np.mean(ori_h[1][mask_h]))) < 0.1

    vertical = np.zeros((21, 21), dtype=bool)
    vertical[3:18, 10] = True
    ori_v, valid_v = orientation_from_skeleton(vertical, radius=3)
    mask_v = valid_v > 0
    assert mask_v.any()
    assert np.mean(ori_v[0][mask_v]) < -0.95
    assert abs(float(np.mean(ori_v[1][mask_v]))) < 0.1


def test_axial_orientation_flip_transform():
    image = np.zeros((1, 4, 5), dtype=np.float32)
    orientation = np.zeros((2, 4, 5), dtype=np.float32)
    orientation[1] = 1.0
    targets = {
        "region": np.zeros((4, 5), np.float32),
        "orientation": orientation,
    }

    _, hflip = apply_geometric_transform(image, targets, hflip=True)
    assert np.allclose(hflip["orientation"][0], 0.0)
    assert np.allclose(hflip["orientation"][1], -1.0)

    _, both = apply_geometric_transform(
        image,
        targets,
        hflip=True,
        vflip=True,
    )
    assert np.allclose(both["orientation"][0], 0.0)
    assert np.allclose(both["orientation"][1], 1.0)
