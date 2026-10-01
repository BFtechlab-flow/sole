import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as cc_label


def _neighbors4(index, h, w):
    y, x = divmod(int(index), w)
    if y > 0:
        yield index - w
    if y + 1 < h:
        yield index + w
    if x > 0:
        yield index - 1
    if x + 1 < w:
        yield index + 1


def h0_persistence_pairs(field):
    """Exact H0 superlevel persistence pairs on a 4-neighbor pixel complex.

    Returns dictionaries with birth/death values and creator/destroyer flat pixel
    indices. The final surviving connected component has death=0 and
    death_index=-1. This is intentionally H0-only: it models fragmentation, the
    dominant topology error for filament instances, and does not claim H1 loops.
    """
    values = np.asarray(field, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("h0_persistence_pairs expects a 2D field")
    h, w = values.shape
    flat = values.ravel()
    # Deterministic tie handling: higher value first, then lower flat index.
    order = np.lexsort((np.arange(flat.size, dtype=np.int64), -flat))
    parent = np.full(flat.size, -1, dtype=np.int64)
    birth_value = np.zeros(flat.size, dtype=np.float32)
    birth_index = np.full(flat.size, -1, dtype=np.int64)
    active = np.zeros(flat.size, dtype=bool)
    pairs = []

    def find_root(i):
        i = int(i)
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    for idx in order:
        idx = int(idx)
        active[idx] = True
        parent[idx] = idx
        birth_value[idx] = flat[idx]
        birth_index[idx] = idx

        roots = []
        for nb in _neighbors4(idx, h, w):
            if active[nb]:
                r = find_root(nb)
                if r not in roots:
                    roots.append(r)
        if not roots:
            continue

        current_root = find_root(idx)
        roots.append(current_root)
        roots = list(dict.fromkeys(find_root(r) for r in roots))
        # Elder rule. Ties prefer the component with lower creator index.
        survivor = max(roots, key=lambda r: (float(birth_value[r]), -int(birth_index[r])))
        for r in roots:
            r = find_root(r)
            if r == survivor:
                continue
            pairs.append(
                {
                    "birth": float(birth_value[r]),
                    "death": float(flat[idx]),
                    "birth_index": int(birth_index[r]),
                    "death_index": idx,
                }
            )
            parent[r] = survivor
        parent[current_root] = survivor

    roots = sorted({find_root(i) for i in np.flatnonzero(active)})
    for r in roots:
        pairs.append(
            {
                "birth": float(birth_value[r]),
                "death": 0.0,
                "birth_index": int(birth_index[r]),
                "death_index": -1,
            }
        )
    pairs.sort(key=lambda p: (p["birth"] - p["death"]), reverse=True)
    return pairs


def _component_count(binary):
    _, n = cc_label(np.asarray(binary, dtype=bool), structure=np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8))
    return int(n)


def h0_persistence_loss(logits, target, max_size=64, max_features=64):
    """Critical-pixel H0 persistent-homology loss.

    The persistent pairs are selected on a detached downsampled probability map,
    then their creator/destroyer pixel values are gathered from the differentiable
    tensor. The number of desired long-lived H0 features comes from connected
    components in the downsampled ground-truth region mask.

    This follows the critical-pixel strategy used by topology-preserving
    segmentation losses while staying dependency-light and auditable.
    """
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("h0_persistence_loss expects Bx1xHxW logits")
    size = min(int(max_size), int(logits.shape[-2]), int(logits.shape[-1]))
    p = torch.sigmoid(logits)
    p_small = F.interpolate(p, size=(size, size), mode="bilinear", align_corners=False)
    t_small = F.interpolate(target.float(), size=(size, size), mode="nearest")

    losses = []
    for b in range(p_small.shape[0]):
        p2 = p_small[b, 0]
        t2 = t_small[b, 0]
        detached = p2.detach().float().cpu().numpy()
        pairs = h0_persistence_pairs(detached)[: int(max_features)]
        desired = _component_count(t2.detach().cpu().numpy() > 0.5)
        desired = min(desired, len(pairs))

        flat = p2.reshape(-1)
        for rank, pair in enumerate(pairs):
            birth = flat[pair["birth_index"]]
            if pair["death_index"] >= 0:
                death = flat[pair["death_index"]]
            else:
                death = birth.new_zeros(())
            if rank < desired:
                # Desired components should have strong persistence: creator high,
                # merger/destruction low.
                losses.append((1.0 - birth).square() + death.square())
            else:
                # Extra components are fragmentation/noise: suppress persistence.
                losses.append((birth - death).square())

        if not pairs and desired > 0:
            losses.append((1.0 - p2.max()).square())

    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()
