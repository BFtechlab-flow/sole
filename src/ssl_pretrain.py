from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class MaskedSolarAutoencoder(nn.Module):
    """MAE-style warm-up whose 4-channel encoder transfers directly into FILA-Net."""

    def __init__(self, encoder_name: str, pretrained: bool = False):
        super().__init__()
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            in_chans=4,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        channels = self.encoder.feature_info.channels()
        c = 96
        self.lat = nn.ModuleList([nn.Conv2d(k, c, 1) for k in channels])
        self.decoder = nn.Sequential(
            nn.Conv2d(c * 4, c, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c, c // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c // 2, 1, 1),
        )

    def forward(self, x):
        feats = self.encoder(x)
        size = feats[0].shape[-2:]
        z = torch.cat(
            [F.interpolate(lat(f), size=size, mode="bilinear", align_corners=False) for lat, f in zip(self.lat, feats)],
            dim=1,
        )
        y = self.decoder(z)
        return F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)


def random_mask(x, patch: int = 24, ratio: float = 0.45):
    b, _, h, w = x.shape
    mask = torch.ones((b, 1, h, w), device=x.device)
    gh = max(1, h // patch)
    gw = max(1, w // patch)
    for bi in range(b):
        ids = torch.randperm(gh * gw, device=x.device)
        n = int(round(ratio * gh * gw))
        for idx in ids[:n]:
            y = int(idx // gw) * patch
            z = int(idx % gw) * patch
            mask[bi, :, y : min(y + patch, h), z : min(z + patch, w)] = 0
    return x * mask, mask


def load_ssl_encoder(model, checkpoint_or_path, strict: bool = True) -> dict:
    if isinstance(checkpoint_or_path, (str, Path)):
        checkpoint = torch.load(checkpoint_or_path, map_location="cpu")
    else:
        checkpoint = checkpoint_or_path
    if int(checkpoint.get("input_channels", -1)) != 4:
        raise ValueError("SSL encoder checkpoint must have input_channels=4")
    if "encoder" not in checkpoint:
        raise KeyError("SSL checkpoint is missing 'encoder'")
    model.encoder.load_state_dict(checkpoint["encoder"], strict=strict)
    return checkpoint
