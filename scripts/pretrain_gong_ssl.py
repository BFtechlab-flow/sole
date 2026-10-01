import argparse
import random
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from src.data import build_input_channels


def gong_urls(center="20111216103234", site="C", minutes=12):
    t0 = datetime.strptime(center, "%Y%m%d%H%M%S")
    urls = []
    for delta in range(-minutes, minutes + 1):
        t = t0 + timedelta(minutes=delta)
        stamp = t.strftime("%Y%m%d%H%M%S")
        ym = t.strftime("%Y%m")
        ymd = t.strftime("%Y%m%d")
        urls.append(f"https://gong2.nso.edu/ftp/HA/has/{ym}/{ymd}/{stamp[:-2]}34{site}h.jpg")
    return urls


def download_many(urls, out_dir, minimum=1):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, url in enumerate(urls):
        path = out_dir / f"gong_{i:03d}.jpg"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "FILA-Net SSL research"})
            with urllib.request.urlopen(req, timeout=12) as response:
                path.write_bytes(response.read())
            if cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) is not None:
                paths.append(path)
        except Exception:
            if path.exists():
                path.unlink()
        time.sleep(0.05)
    if len(paths) < minimum:
        raise RuntimeError(f"only {len(paths)} GONG images downloaded; need >= {minimum}")
    return paths


def random_mask(x, patch=24, ratio=0.45):
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


class MaskedSolarAutoencoder(nn.Module):
    """Small MAE-style pretrainer whose encoder weights transfer into FILA-Net."""

    def __init__(self, encoder_name, pretrained=False):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="convnext_tiny.fb_in1k")
    ap.add_argument("--imagenet-pretrained", action="store_true")
    ap.add_argument("--output", default="artifacts/gong_ssl_encoder.pt")
    ap.add_argument("--cache-dir", default="artifacts/gong_ssl_images")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps-per-epoch", type=int, default=12)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--minutes", type=int, default=10)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    torch.manual_seed(2026)
    random.seed(2026)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    urls = gong_urls(minutes=args.minutes)
    if args.max_images is not None:
        urls = urls[: max(1, int(args.max_images))]
    paths = download_many(urls, Path(args.cache_dir), minimum=1)
    arrays = []
    for path in paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        gray = cv2.resize(gray, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
        arrays.append(build_input_channels(gray))

    model = MaskedSolarAutoencoder(args.encoder, pretrained=args.imagenet_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    model.train()
    history = []
    for epoch in range(args.epochs):
        running = 0.0
        for _ in range(args.steps_per_epoch):
            img = arrays[random.randrange(len(arrays))]
            x = torch.from_numpy(img[None]).float().to(device)
            masked, keep = random_mask(x)
            pred = model(masked)
            target_raw = x[:, :1]
            missing = 1.0 - keep
            loss_missing = (torch.abs(pred - target_raw) * missing).sum() / (missing.sum() + 1e-6)
            loss_context = 0.10 * (torch.abs(pred - target_raw) * keep).sum() / (keep.sum() + 1e-6)
            loss = loss_missing + loss_context
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item())
        avg = running / args.steps_per_epoch
        history.append(avg)
        print(f"ssl epoch={epoch + 1}/{args.epochs} loss={avg:.6f} images={len(arrays)} device={device}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "encoder_name": args.encoder,
            "input_channels": 4,
            "source": "public GONG H-alpha small JPEGs; no external ground-truth labels",
            "images": len(arrays),
            "loss_history": history,
        },
        out,
    )
    print(out)


if __name__ == "__main__":
    main()
