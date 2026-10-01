import torch

from src.losses import cldice_loss, orientation_loss


def test_cldice_penalizes_broken_centerline_more_than_complete_one():
    target = torch.zeros(1, 1, 32, 32)
    target[:, :, 16, 4:28] = 1.0

    complete = torch.full_like(target, -8.0)
    complete[target > 0] = 8.0

    broken = complete.clone()
    broken[:, :, 16, 14:18] = -8.0

    good_loss = cldice_loss(complete, target, iterations=4)
    broken_loss = cldice_loss(broken, target, iterations=4)
    assert good_loss < broken_loss


def test_orientation_loss_is_low_for_matching_axial_vectors():
    pred = torch.tensor([[[[1.0]], [[0.0]]]])
    target = torch.tensor([[[[1.0]], [[0.0]]]])
    valid = torch.ones(1, 1, 1, 1)
    loss = orientation_loss(pred, target, valid)
    assert float(loss) < 1e-6
