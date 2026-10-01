import numpy as np
from scipy.ndimage import convolve, distance_transform_edt
from skimage.draw import line
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


def bridge_skeleton_fragments(
    skeleton,
    region_prob=None,
    boundary_prob=None,
    orientation=None,
    max_distance=28,
    min_alignment=0.80,
    min_region=0.20,
    max_boundary=0.75,
):
    """Connect compatible skeleton endpoints using a scored endpoint graph."""
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
        endpoint_info.append((idx, int(y), int(x), comp, tangent))

    candidates = []
    for a in range(len(endpoint_info)):
        ia, ya, xa, ca, ta = endpoint_info[a]
        for b in range(a + 1, len(endpoint_info)):
            ib, yb, xb, cb, tb = endpoint_info[b]
            if ca == cb:
                continue

            dx, dy = float(xb - xa), float(yb - ya)
            dist = float(np.hypot(dx, dy))
            if dist <= 1e-6 or dist > max_distance:
                continue
            line_unit = np.array([dx / dist, dy / dist], np.float32)

            align_a = _axial_alignment(orientation, ya, xa, line_unit)
            align_b = _axial_alignment(orientation, yb, xb, line_unit)
            if align_a is None:
                align_a = abs(float(np.dot(ta, line_unit))) if ta is not None else 0.0
            if align_b is None:
                align_b = abs(float(np.dot(tb, line_unit))) if tb is not None else 0.0
            if min(align_a, align_b) < min_alignment:
                continue

            rr, cc = line(ya, xa, yb, xb)
            region_mean = (
                float(np.mean(region_prob[rr, cc])) if region_prob is not None else 1.0
            )
            boundary_mean = (
                float(np.mean(boundary_prob[rr, cc])) if boundary_prob is not None else 0.0
            )
            if region_mean < min_region or boundary_mean > max_boundary:
                continue

            score = (
                dist / max(max_distance, 1e-6)
                + 0.5 * ((1.0 - align_a) + (1.0 - align_b))
                + 0.5 * (1.0 - region_mean)
                + 0.5 * boundary_mean
            )
            candidates.append(
                (score, ia, ib, ca, cb, rr, cc, dist, align_a, align_b)
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
    for score, ia, ib, ca, cb, rr, cc, dist, aa, ab in candidates:
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
                "alignment_a": float(aa),
                "alignment_b": float(ab),
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
    region_threshold=0.45,
    center_threshold=0.35,
    min_region_area=64,
    min_instance_area=250,
    bridge_max_distance=28,
    bridge_min_alignment=0.80,
    bridge_min_region=0.20,
    bridge_max_boundary=0.75,
    **kwargs,
):
    fg = remove_small_components(region > region_threshold, min_region_area)
    seeds = skeletonize(
        closing((center > center_threshold) & fg, footprint=disk(1))
    )

    if seeds.any() and bridge_max_distance > 0:
        seeds, bridge_mask, _ = bridge_skeleton_fragments(
            seeds,
            region_prob=region,
            boundary_prob=boundary,
            orientation=orientation,
            max_distance=bridge_max_distance,
            min_alignment=bridge_min_alignment,
            min_region=bridge_min_region,
            max_boundary=bridge_max_boundary,
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
