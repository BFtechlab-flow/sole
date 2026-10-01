import numpy as np
from scipy.ndimage import convolve, distance_transform_edt
from skimage.draw import line
from skimage.graph import route_through_array
from skimage.measure import label
from skimage.morphology import closing, dilation, disk, skeletonize
from skimage.segmentation import watershed


def remove_small_components(mask, min_area):
    """Remove connected components smaller than min_area without deprecated APIs."""
    labels = label(mask, connectivity=2)
    if labels.max() == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labels.ravel())
    keep = counts >= int(min_area)
    keep[0] = False
    return keep[labels]


def find_endpoints(skeleton):
    sk = np.asarray(skeleton, dtype=bool)
    neighbors = convolve(
        sk.astype(np.uint8),
        np.ones((3, 3), np.uint8),
        mode="constant",
        cval=0,
    ) - sk.astype(np.uint8)
    return sk & (neighbors == 1)


def _local_tangent(component_mask, y, x, radius=5):
    h, w = component_mask.shape
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    yy, xx = np.nonzero(component_mask[y0:y1, x0:x1])
    if len(xx) < 2:
        return None
    pts = np.column_stack((xx + x0, yy + y0)).astype(np.float32)
    pts -= pts.mean(axis=0, keepdims=True)
    cov = pts.T @ pts
    vals, vecs = np.linalg.eigh(cov)
    vec = vecs[:, int(np.argmax(vals))]
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return None
    return vec / norm


def _axial_alignment(orientation, y, x, line_unit):
    if orientation is None:
        return None
    vec = orientation[:, y, x]
    norm = float(np.linalg.norm(vec))
    if norm < 0.25:
        return None
    vec = vec / norm
    theta = np.arctan2(line_unit[1], line_unit[0])
    line_axial = np.array([np.cos(2 * theta), np.sin(2 * theta)], np.float32)
    cos2delta = float(np.clip(np.dot(vec, line_axial), -1.0, 1.0))
    return float(np.sqrt(max(0.0, (1.0 + cos2delta) * 0.5)))


def _local_scalar(field, y, x, radius=2):
    if field is None:
        return None
    h, w = field.shape
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    values = np.asarray(field[y0:y1, x0:x1], dtype=np.float32)
    values = values[np.isfinite(values) & (values > 1e-4)]
    return float(np.median(values)) if values.size else None


def _local_embedding(field, y, x, radius=2):
    if field is None:
        return None
    emb = np.asarray(field, dtype=np.float32)
    if emb.ndim != 3:
        raise ValueError("instance_embedding must be CxHxW")
    _, h, w = emb.shape
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    values = emb[:, y0:y1, x0:x1].reshape(emb.shape[0], -1)
    valid = np.linalg.norm(values, axis=0) > 1e-5
    if not np.any(valid):
        return None
    vec = np.mean(values[:, valid], axis=1)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-6 else None


def _unit_from_path(rr, cc, from_start=True, span=3):
    if len(rr) < 2:
        return None
    if from_start:
        j = min(int(span), len(rr) - 1)
        dx, dy = float(cc[j] - cc[0]), float(rr[j] - rr[0])
    else:
        j = max(0, len(rr) - 1 - int(span))
        dx, dy = float(cc[-1] - cc[j]), float(rr[-1] - rr[j])
    norm = float(np.hypot(dx, dy))
    if norm < 1e-6:
        return None
    return np.array([dx / norm, dy / norm], np.float32)


def _geodesic_bridge_path(
    ya,
    xa,
    yb,
    xb,
    region_prob=None,
    boundary_prob=None,
    padding=12,
    region_weight=2.0,
    boundary_weight=3.0,
):
    """Minimum-cost endpoint path using region support and boundary avoidance."""
    if region_prob is None and boundary_prob is None:
        rr, cc = line(ya, xa, yb, xb)
        return rr.astype(int), cc.astype(int), float(np.hypot(yb - ya, xb - xa))

    ref = region_prob if region_prob is not None else boundary_prob
    h, w = ref.shape
    y0 = max(0, min(ya, yb) - int(padding))
    y1 = min(h, max(ya, yb) + int(padding) + 1)
    x0 = max(0, min(xa, xb) - int(padding))
    x1 = min(w, max(xa, xb) + int(padding) + 1)
    cost = np.ones((y1 - y0, x1 - x0), np.float32)
    if region_prob is not None:
        region = np.clip(region_prob[y0:y1, x0:x1], 0.0, 1.0)
        cost += float(region_weight) * (1.0 - region)
    if boundary_prob is not None:
        boundary = np.clip(boundary_prob[y0:y1, x0:x1], 0.0, 1.0)
        cost += float(boundary_weight) * boundary

    start = (int(ya - y0), int(xa - x0))
    end = (int(yb - y0), int(xb - x0))
    path, _ = route_through_array(cost, start, end, fully_connected=True, geometric=True)
    path = np.asarray(path, dtype=np.int32)
    rr = path[:, 0] + y0
    cc = path[:, 1] + x0
    if len(path) > 1:
        steps = np.diff(np.column_stack((rr, cc)).astype(np.float32), axis=0)
        path_length = float(np.linalg.norm(steps, axis=1).sum())
    else:
        path_length = 0.0
    return rr, cc, path_length


def bridge_skeleton_fragments(
    skeleton,
    region_prob=None,
    boundary_prob=None,
    orientation=None,
    width=None,
    instance_embedding=None,
    max_distance=28,
    min_alignment=0.80,
    min_region=0.20,
    max_boundary=0.75,
    max_width_ratio=2.5,
    width_weight=0.35,
    geodesic=True,
    geodesic_padding=12,
    geodesic_region_weight=2.0,
    geodesic_boundary_weight=3.0,
    max_geodesic_ratio=1.8,
    min_embedding_similarity=0.15,
    embedding_weight=0.35,
):
    """Connect compatible endpoints using geometry, appearance and learned identity."""
    sk = np.asarray(skeleton, dtype=bool)
    components = label(sk, connectivity=2)
    endpoints = np.argwhere(find_endpoints(sk))
    bridges = np.zeros_like(sk, dtype=bool)

    if len(endpoints) < 2 or components.max() < 2 or max_distance <= 0:
        return sk.copy(), bridges, []

    endpoint_info = []
    for idx, (y, x) in enumerate(endpoints):
        comp = int(components[y, x])
        tangent = _local_tangent(components == comp, int(y), int(x))
        local_width = _local_scalar(width, int(y), int(x))
        local_embedding = _local_embedding(instance_embedding, int(y), int(x))
        endpoint_info.append((idx, int(y), int(x), comp, tangent, local_width, local_embedding))

    candidates = []
    for a in range(len(endpoint_info)):
        ia, ya, xa, ca, ta, wa, ea = endpoint_info[a]
        for b in range(a + 1, len(endpoint_info)):
            ib, yb, xb, cb, tb, wb, eb = endpoint_info[b]
            if ca == cb:
                continue

            dx, dy = float(xb - xa), float(yb - ya)
            dist = float(np.hypot(dx, dy))
            if dist <= 1e-6 or dist > max_distance:
                continue

            width_ratio = 1.0
            width_penalty = 0.0
            if wa is not None and wb is not None:
                width_ratio = max(wa, wb) / max(min(wa, wb), 1e-4)
                if width_ratio > max_width_ratio:
                    continue
                width_penalty = abs(float(np.log(max(wa, 1e-4) / max(wb, 1e-4))))

            if geodesic:
                rr, cc, path_length = _geodesic_bridge_path(
                    ya,
                    xa,
                    yb,
                    xb,
                    region_prob=region_prob,
                    boundary_prob=boundary_prob,
                    padding=geodesic_padding,
                    region_weight=geodesic_region_weight,
                    boundary_weight=geodesic_boundary_weight,
                )
            else:
                rr, cc = line(ya, xa, yb, xb)
                path_length = dist
            geodesic_ratio = path_length / max(dist, 1e-6)
            if geodesic_ratio > max_geodesic_ratio:
                continue

            start_unit = _unit_from_path(rr, cc, True)
            end_unit = _unit_from_path(rr, cc, False)
            if start_unit is None or end_unit is None:
                continue
            align_a = _axial_alignment(orientation, ya, xa, start_unit)
            align_b = _axial_alignment(orientation, yb, xb, end_unit)
            if align_a is None:
                align_a = abs(float(np.dot(ta, start_unit))) if ta is not None else 0.0
            if align_b is None:
                align_b = abs(float(np.dot(tb, end_unit))) if tb is not None else 0.0
            if min(align_a, align_b) < min_alignment:
                continue

            region_mean = float(np.mean(region_prob[rr, cc])) if region_prob is not None else 1.0
            boundary_mean = float(np.mean(boundary_prob[rr, cc])) if boundary_prob is not None else 0.0
            if region_mean < min_region or boundary_mean > max_boundary:
                continue

            embedding_similarity = None
            embedding_penalty = 0.0
            if ea is not None and eb is not None:
                embedding_similarity = float(np.clip(np.dot(ea, eb), -1.0, 1.0))
                if embedding_similarity < min_embedding_similarity:
                    continue
                embedding_penalty = 1.0 - embedding_similarity

            score = (
                path_length / max(max_distance, 1e-6)
                + 0.5 * ((1.0 - align_a) + (1.0 - align_b))
                + 0.5 * (1.0 - region_mean)
                + 0.5 * boundary_mean
                + width_weight * width_penalty
                + embedding_weight * embedding_penalty
            )
            candidates.append(
                (
                    score,
                    ia,
                    ib,
                    ca,
                    cb,
                    rr,
                    cc,
                    dist,
                    path_length,
                    geodesic_ratio,
                    align_a,
                    align_b,
                    width_ratio,
                    embedding_similarity,
                )
            )

    candidates.sort(key=lambda item: item[0])
    parent = {c: c for c in range(1, int(components.max()) + 1)}

    def find_root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find_root(a), find_root(b)
        if ra != rb:
            parent[rb] = ra

    used_endpoints = set()
    accepted = []
    for item in candidates:
        (
            score,
            ia,
            ib,
            ca,
            cb,
            rr,
            cc,
            dist,
            path_length,
            geodesic_ratio,
            aa,
            ab,
            wr,
            emb_sim,
        ) = item
        if ia in used_endpoints or ib in used_endpoints:
            continue
        if find_root(ca) == find_root(cb):
            continue
        bridges[rr, cc] = True
        used_endpoints.update((ia, ib))
        union(ca, cb)
        accepted.append(
            {
                "distance": float(dist),
                "path_length": float(path_length),
                "geodesic_ratio": float(geodesic_ratio),
                "alignment_a": float(aa),
                "alignment_b": float(ab),
                "width_ratio": float(wr),
                "embedding_similarity": None if emb_sim is None else float(emb_sim),
                "score": float(score),
            }
        )

    return sk | bridges, bridges, accepted


def reconstruct_instances(
    region,
    center,
    boundary,
    distance,
    orientation=None,
    width=None,
    instance_embedding=None,
    region_threshold=0.45,
    center_threshold=0.35,
    min_region_area=64,
    min_instance_area=250,
    bridge_max_distance=28,
    bridge_min_alignment=0.80,
    bridge_min_region=0.20,
    bridge_max_boundary=0.75,
    bridge_max_width_ratio=2.5,
    bridge_width_weight=0.35,
    bridge_geodesic=True,
    bridge_geodesic_padding=12,
    bridge_geodesic_region_weight=2.0,
    bridge_geodesic_boundary_weight=3.0,
    bridge_max_geodesic_ratio=1.8,
    bridge_min_embedding_similarity=0.15,
    bridge_embedding_weight=0.35,
    **kwargs,
):
    fg = remove_small_components(region > region_threshold, min_region_area)
    seeds = skeletonize(closing((center > center_threshold) & fg, footprint=disk(1)))

    if seeds.any() and bridge_max_distance > 0:
        seeds, bridge_mask, _ = bridge_skeleton_fragments(
            seeds,
            region_prob=region,
            boundary_prob=boundary,
            orientation=orientation,
            width=width,
            instance_embedding=instance_embedding,
            max_distance=bridge_max_distance,
            min_alignment=bridge_min_alignment,
            min_region=bridge_min_region,
            max_boundary=bridge_max_boundary,
            max_width_ratio=bridge_max_width_ratio,
            width_weight=bridge_width_weight,
            geodesic=bridge_geodesic,
            geodesic_padding=bridge_geodesic_padding,
            geodesic_region_weight=bridge_geodesic_region_weight,
            geodesic_boundary_weight=bridge_geodesic_boundary_weight,
            max_geodesic_ratio=bridge_max_geodesic_ratio,
            min_embedding_similarity=bridge_min_embedding_similarity,
            embedding_weight=bridge_embedding_weight,
        )
        if bridge_mask.any():
            fg = fg | dilation(bridge_mask, footprint=disk(1))
            seeds = skeletonize(closing(seeds, footprint=disk(1)))

    markers = label(seeds, connectivity=2)
    if markers.max() == 0 and fg.any():
        d = distance_transform_edt(fg)
        markers = label(d > np.percentile(d[fg], 80), connectivity=2)

    inst = watershed(
        (-distance + 0.9 * boundary).astype(np.float32),
        markers=markers,
        mask=fg,
    )
    return [
        (inst == i).astype(np.uint8)
        for i in range(1, inst.max() + 1)
        if (inst == i).sum() >= min_instance_area
    ]
