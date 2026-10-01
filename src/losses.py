import torch
import torch.nn.functional as F


def dice(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((2, 3))
    den = p.sum((2, 3)) + target.sum((2, 3))
    return (1 - (2 * inter + eps) / (den + eps)).mean()


def soft_erode(x):
    p1 = -F.max_pool2d(-x, (3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-x, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(p1, p2)


def soft_dilate(x):
    return F.max_pool2d(x, 3, stride=1, padding=1)


def soft_open(x):
    return soft_dilate(soft_erode(x))


def soft_skeletonize(x, iterations=10):
    """Differentiable morphological skeleton used by clDice."""
    img = x
    opened = soft_open(img)
    skel = F.relu(img - opened)
    for _ in range(iterations):
        img = soft_erode(img)
        opened = soft_open(img)
        delta = F.relu(img - opened)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cldice_loss(logits, target, iterations=10, eps=1e-6):
    pred = torch.sigmoid(logits)
    skel_pred = soft_skeletonize(pred, iterations)
    skel_true = soft_skeletonize(target, iterations)

    dims = (2, 3)
    tprec = ((skel_pred * target).sum(dims) + eps) / (
        skel_pred.sum(dims) + eps
    )
    tsens = ((skel_true * pred).sum(dims) + eps) / (
        skel_true.sum(dims) + eps
    )
    cldice = (2.0 * tprec * tsens + eps) / (tprec + tsens + eps)
    return (1.0 - cldice).mean()


def orientation_loss(pred, target, valid, eps=1e-6):
    """Axial cosine loss for [cos(2θ), sin(2θ)] orientation vectors."""
    pred = F.normalize(pred, dim=1, eps=eps)
    target = F.normalize(target, dim=1, eps=eps)
    dot = (pred * target).sum(dim=1, keepdim=True)
    mask = valid.float()
    denom = mask.sum()
    return ((1.0 - dot) * mask).sum() / (denom + eps)


def total_loss(pred, t, w):
    rb = F.binary_cross_entropy_with_logits(pred["region"], t["region"])
    rd = dice(pred["region"], t["region"])
    cb = F.binary_cross_entropy_with_logits(pred["centerline"], t["centerline"])
    cd = dice(pred["centerline"], t["centerline"])
    bb = F.binary_cross_entropy_with_logits(pred["boundary"], t["boundary"])
    dl = F.smooth_l1_loss(torch.sigmoid(pred["distance"]), t["distance"])

    loss = (
        w["region_bce"] * rb
        + w["region_dice"] * rd
        + w["center_bce"] * cb
        + w["center_dice"] * cd
        + w["boundary_bce"] * bb
        + w["distance"] * dl
    )

    if w.get("topology", 0.0) > 0:
        loss = loss + w["topology"] * cldice_loss(pred["region"], t["region"])

    if w.get("orientation", 0.0) > 0 and "orientation" in pred:
        loss = loss + w["orientation"] * orientation_loss(
            pred["orientation"],
            t["orientation"],
            t["orientation_valid"],
        )
    return loss
