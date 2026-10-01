import numpy as np
import torch
import torch.nn.functional as F

SCALAR_KEYS = ("region", "centerline", "boundary", "distance")
OPTIONAL_PROB_KEYS = ("width", "curvature", "endpoint", "junction")


def positions(n, tile, overlap):
    if n <= tile:
        return [0]
    step = tile - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than tile")
    p = list(range(0, n - tile + 1, step))
    if p[-1] != n - tile:
        p.append(n - tile)
    return p


def _inverse_flip(tensor, hflip=False, vflip=False):
    if hflip:
        tensor = torch.flip(tensor, dims=(-1,))
    if vflip:
        tensor = torch.flip(tensor, dims=(-2,))
    return tensor


def _inverse_orientation(orientation, hflip=False, vflip=False):
    orientation = _inverse_flip(orientation, hflip=hflip, vflip=vflip)
    if hflip ^ vflip:
        orientation = orientation.clone()
        orientation[:, 1] *= -1.0
    return orientation


@torch.no_grad()
def _predict_patch(model, tensor, tta=True):
    transforms = [(False, False)]
    if tta:
        transforms.extend([(True, False), (False, True), (True, True)])

    summed = {k: None for k in SCALAR_KEYS}
    optional_summed = {}
    orientation_sum = None
    uncertainty_sum = None
    embedding_sum = None

    for hflip, vflip in transforms:
        aug = _inverse_flip(tensor, hflip=hflip, vflip=vflip)
        raw = model(aug)

        for k in SCALAR_KEYS:
            pred = torch.sigmoid(raw[k])
            pred = _inverse_flip(pred, hflip=hflip, vflip=vflip)
            summed[k] = pred if summed[k] is None else summed[k] + pred

        for k in OPTIONAL_PROB_KEYS:
            if k not in raw:
                continue
            pred = torch.sigmoid(raw[k])
            pred = _inverse_flip(pred, hflip=hflip, vflip=vflip)
            optional_summed[k] = pred if k not in optional_summed else optional_summed[k] + pred

        if "uncertainty" in raw:
            unc = torch.sigmoid(raw["uncertainty"])
            unc = _inverse_flip(unc, hflip=hflip, vflip=vflip)
            uncertainty_sum = unc if uncertainty_sum is None else uncertainty_sum + unc

        if "orientation" in raw:
            ori = F.normalize(raw["orientation"], dim=1, eps=1e-6)
            ori = _inverse_orientation(ori, hflip=hflip, vflip=vflip)
            orientation_sum = ori if orientation_sum is None else orientation_sum + ori

        if "instance_embedding" in raw:
            emb = F.normalize(raw["instance_embedding"], dim=1, eps=1e-6)
            emb = _inverse_flip(emb, hflip=hflip, vflip=vflip)
            embedding_sum = emb if embedding_sum is None else embedding_sum + emb

    scale = 1.0 / len(transforms)
    out = {k: (v * scale)[0, 0].cpu().numpy() for k, v in summed.items()}
    out.update({k: (v * scale)[0, 0].cpu().numpy() for k, v in optional_summed.items()})
    if uncertainty_sum is not None:
        out["uncertainty"] = (uncertainty_sum * scale)[0, 0].cpu().numpy()
    if orientation_sum is not None:
        ori = F.normalize(orientation_sum * scale, dim=1, eps=1e-6)
        out["orientation"] = ori[0].cpu().numpy()
    if embedding_sum is not None:
        emb = F.normalize(embedding_sum * scale, dim=1, eps=1e-6)
        out["instance_embedding"] = emb[0].cpu().numpy()
    return out


@torch.no_grad()
def predict_tiled(model, x, device, tile=1024, overlap=256, tta=True):
    _, h, w = x.shape
    acc = {k: np.zeros((h, w), np.float32) for k in SCALAR_KEYS}
    optional_acc = {}
    orientation_acc = np.zeros((2, h, w), np.float32)
    has_orientation = False
    embedding_acc = None
    weight = np.zeros((h, w), np.float32)
    win = np.maximum(np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32), 0.05)

    for y in positions(h, tile, overlap):
        for z in positions(w, tile, overlap):
            p = x[:, y : y + tile, z : z + tile]
            tensor = torch.from_numpy(p[None]).float().to(device)
            out = _predict_patch(model, tensor, tta=tta)
            hh, ww = p.shape[-2:]
            q = win[:hh, :ww]

            for k in SCALAR_KEYS:
                acc[k][y : y + hh, z : z + ww] += out[k] * q

            for k in OPTIONAL_PROB_KEYS + ("uncertainty",):
                if k not in out:
                    continue
                if k not in optional_acc:
                    optional_acc[k] = np.zeros((h, w), np.float32)
                optional_acc[k][y : y + hh, z : z + ww] += out[k] * q

            if "orientation" in out:
                has_orientation = True
                orientation_acc[:, y : y + hh, z : z + ww] += out["orientation"] * q[None]

            if "instance_embedding" in out:
                if embedding_acc is None:
                    embedding_acc = np.zeros((out["instance_embedding"].shape[0], h, w), np.float32)
                embedding_acc[:, y : y + hh, z : z + ww] += out["instance_embedding"] * q[None]

            weight[y : y + hh, z : z + ww] += q

    result = {k: v / np.maximum(weight, 1e-6) for k, v in acc.items()}
    result.update({k: v / np.maximum(weight, 1e-6) for k, v in optional_acc.items()})
    if has_orientation:
        orientation = orientation_acc / np.maximum(weight[None], 1e-6)
        norm = np.linalg.norm(orientation, axis=0, keepdims=True)
        result["orientation"] = orientation / np.maximum(norm, 1e-6)
    if embedding_acc is not None:
        embedding = embedding_acc / np.maximum(weight[None], 1e-6)
        norm = np.linalg.norm(embedding, axis=0, keepdims=True)
        result["instance_embedding"] = embedding / np.maximum(norm, 1e-6)
    return result
