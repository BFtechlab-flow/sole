import torch
import torch.nn.functional as F

from .topology import h0_persistence_loss


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
    """Differentiable morphological skeleton used by topology-aware losses."""
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
    tprec = ((skel_pred * target).sum(dims) + eps) / (skel_pred.sum(dims) + eps)
    tsens = ((skel_true * pred).sum(dims) + eps) / (skel_true.sum(dims) + eps)
    cldice = (2.0 * tprec * tsens + eps) / (tprec + tsens + eps)
    return (1.0 - cldice).mean()


def centerline_ce_loss(logits, target, iterations=8):
    """CE focused on topologically critical centerline pixels."""
    base = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pred_skel = soft_skeletonize(torch.sigmoid(logits), iterations).detach()
    true_skel = soft_skeletonize(target, iterations)
    critical = torch.clamp(pred_skel + true_skel, 0.0, 1.0)
    return (base * (1.0 + 4.0 * critical)).mean()


def orientation_loss(pred, target, valid, eps=1e-6):
    """Axial cosine loss for [cos(2theta), sin(2theta)] orientation vectors."""
    pred = F.normalize(pred, dim=1, eps=eps)
    target = F.normalize(target, dim=1, eps=eps)
    dot = (pred * target).sum(dim=1, keepdim=True)
    mask = valid.float()
    denom = mask.sum()
    return ((1.0 - dot) * mask).sum() / (denom + eps)


def masked_smooth_l1(logits, target, valid, eps=1e-6):
    pred = torch.sigmoid(logits)
    loss = F.smooth_l1_loss(pred, target, reduction="none") * valid.float()
    return loss.sum() / (valid.sum() + eps)


def sparse_focal_bce(logits, target, gamma=2.0, alpha=0.75):
    """Stable focal BCE for sparse endpoints/junctions."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * target + (1.0 - p) * (1.0 - target)
    a = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (a * (1.0 - pt).pow(gamma) * bce).mean()


def radius_balance_loss(center_logits, center_target, width_target, width_valid, eps=1e-6):
    """Give thin and thick filament centerlines comparable influence."""
    bce = F.binary_cross_entropy_with_logits(center_logits, center_target, reduction="none")
    valid = width_valid.float()
    inv_radius = 1.0 / (0.08 + width_target.detach())
    weights = valid * inv_radius
    return (bce * weights).sum() / (weights.sum() + eps)


def uncertainty_region_loss(region_logits, target, uncertainty_raw):
    """Heteroscedastic region loss for annotation/image ambiguity."""
    pixel_bce = F.binary_cross_entropy_with_logits(region_logits, target, reduction="none")
    log_var = torch.clamp(uncertainty_raw, -4.0, 4.0)
    precision = torch.exp(-log_var)
    loss = precision * pixel_bce + 0.5 * log_var + 0.0025 * log_var.square()
    return loss.mean()


def discriminative_instance_loss(
    embedding,
    instance_id,
    delta_var=0.5,
    delta_dist=1.5,
    reg_weight=1e-3,
    max_pixels_per_instance=2048,
):
    """Proposal-free discriminative embedding loss for filament instances.

    Pixels from the same GT filament are pulled toward an instance mean; means
    from different filaments are pushed apart. Background id=0 is ignored.
    Pixel sampling is deterministic and caps memory on 1024x1024 patches.
    """
    if embedding.ndim != 4 or instance_id.ndim != 4:
        raise ValueError("embedding and instance_id must be BxCxHxW and Bx1xHxW")
    if instance_id.shape[1] != 1 or embedding.shape[-2:] != instance_id.shape[-2:]:
        raise ValueError("instance_id shape must match embedding spatial dimensions")

    batch_losses = []
    for b in range(embedding.shape[0]):
        emb = embedding[b]
        ids = instance_id[b, 0].long()
        unique_ids = torch.unique(ids)
        unique_ids = unique_ids[unique_ids > 0]
        if unique_ids.numel() == 0:
            batch_losses.append(embedding[b].sum() * 0.0)
            continue

        centers = []
        var_terms = []
        for iid in unique_ids:
            coords = torch.nonzero(ids == iid, as_tuple=False)
            if coords.shape[0] > max_pixels_per_instance:
                pick = torch.linspace(
                    0,
                    coords.shape[0] - 1,
                    steps=max_pixels_per_instance,
                    device=coords.device,
                ).long()
                coords = coords[pick]
            vectors = emb[:, coords[:, 0], coords[:, 1]].transpose(0, 1)
            center = vectors.mean(dim=0)
            centers.append(center)
            distances = torch.linalg.vector_norm(vectors - center[None], dim=1)
            var_terms.append(F.relu(distances - float(delta_var)).square().mean())

        centers = torch.stack(centers, dim=0)
        var_loss = torch.stack(var_terms).mean()
        if centers.shape[0] > 1:
            d = torch.cdist(centers, centers)
            mask = ~torch.eye(centers.shape[0], dtype=torch.bool, device=d.device)
            dist_loss = F.relu(2.0 * float(delta_dist) - d[mask]).square().mean()
        else:
            dist_loss = centers.sum() * 0.0
        reg_loss = torch.linalg.vector_norm(centers, dim=1).mean()
        batch_losses.append(var_loss + dist_loss + float(reg_weight) * reg_loss)

    return torch.stack(batch_losses).mean()


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

    if w.get("h0_persistence", 0.0) > 0:
        loss = loss + w["h0_persistence"] * h0_persistence_loss(
            pred["region"],
            t["region"],
            max_size=int(w.get("h0_max_size", 64)),
            max_features=int(w.get("h0_max_features", 64)),
        )

    if w.get("orientation", 0.0) > 0 and "orientation" in pred:
        loss = loss + w["orientation"] * orientation_loss(
            pred["orientation"], t["orientation"], t["orientation_valid"]
        )

    if w.get("width", 0.0) > 0 and "width" in pred and "width" in t:
        loss = loss + w["width"] * masked_smooth_l1(
            pred["width"], t["width"], t["width_valid"]
        )

    if w.get("curvature", 0.0) > 0 and "curvature" in pred and "curvature" in t:
        loss = loss + w["curvature"] * masked_smooth_l1(
            pred["curvature"], t["curvature"], t["curvature_valid"]
        )

    if w.get("endpoint", 0.0) > 0 and "endpoint" in pred:
        loss = loss + w["endpoint"] * sparse_focal_bce(pred["endpoint"], t["endpoint"])

    if w.get("junction", 0.0) > 0 and "junction" in pred:
        loss = loss + w["junction"] * sparse_focal_bce(pred["junction"], t["junction"])

    if w.get("centerline_ce", 0.0) > 0:
        loss = loss + w["centerline_ce"] * centerline_ce_loss(
            pred["region"], t["region"]
        )

    if w.get("radius_balance", 0.0) > 0 and "width" in t:
        loss = loss + w["radius_balance"] * radius_balance_loss(
            pred["centerline"], t["centerline"], t["width"], t["width_valid"]
        )

    if w.get("uncertainty_region", 0.0) > 0 and "uncertainty" in pred:
        loss = loss + w["uncertainty_region"] * uncertainty_region_loss(
            pred["region"], t["region"], pred["uncertainty"]
        )

    if w.get("instance_embedding", 0.0) > 0 and "instance_embedding" in pred:
        loss = loss + w["instance_embedding"] * discriminative_instance_loss(
            pred["instance_embedding"],
            t["instance_id"],
            delta_var=float(w.get("embedding_delta_var", 0.5)),
            delta_dist=float(w.get("embedding_delta_dist", 1.5)),
            max_pixels_per_instance=int(w.get("embedding_max_pixels", 2048)),
        )

    return loss
