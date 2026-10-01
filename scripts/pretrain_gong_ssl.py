import argparse
import json
import random
from pathlib import Path
from urllib.parse import urlparse

import cv2
import torch

from src.data import build_input_channels
from src.gong import (
    discover_halpha_urls,
    download_url,
    halpha_metadata,
    load_halpha_uint8,
    select_diverse_halpha_urls,
)
from src.ssl_pretrain import MaskedSolarAutoencoder, random_mask


def download_many(urls, out_dir, minimum=1):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for url in urls:
        name = Path(urlparse(url).path).name
        path = out_dir / name
        try:
            if not path.exists():
                download_url(url, path, retries=2, timeout=30)
            _ = load_halpha_uint8(path)
            paths.append(path)
        except Exception as exc:
            print(f"skip {url}: {exc}")
            if path.exists():
                path.unlink()
    if len(paths) < minimum:
        raise RuntimeError(f"only {len(paths)} NOAA GONG images downloaded; need >= {minimum}")
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="convnext_tiny.fb_in1k")
    ap.add_argument("--imagenet-pretrained", action="store_true")
    ap.add_argument("--output", default="artifacts/gong_ssl_encoder.pt")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--cache-dir", default="artifacts/gong_ssl_images")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps-per-epoch", type=int, default=12)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--minutes", type=int, default=3, help="Legacy compatibility; controls image count when --max-images is omitted")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--min-images", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    count = int(args.max_images) if args.max_images is not None else max(1, 2 * int(args.minutes) + 1)
    all_urls = discover_halpha_urls(limit=None)
    urls = select_diverse_halpha_urls(all_urls, count)
    if not urls:
        raise RuntimeError("NOAA GONG corpus selection returned zero frames")
    print("selected GONG frames:")
    for url in urls:
        meta = halpha_metadata(url)
        print(f"  {meta['timestamp']} site={meta['site']} {url}")

    paths = download_many(urls, Path(args.cache_dir), minimum=min(args.min_images, count))
    arrays = []
    valid_urls = []
    path_by_name = {p.name: p for p in paths}
    for url in urls:
        name = Path(urlparse(url).path).name
        if name not in path_by_name:
            continue
        gray = load_halpha_uint8(path_by_name[name])
        gray = cv2.resize(gray, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
        arrays.append(build_input_channels(gray))
        valid_urls.append(url)

    initialization = "imagenet" if args.imagenet_pretrained else "random"
    print(f"SSL encoder initialization={initialization}")
    model = MaskedSolarAutoencoder(args.encoder, pretrained=args.imagenet_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    step_history = []
    start_epoch = 0
    completed_steps = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if checkpoint.get("encoder_name") != args.encoder:
            raise ValueError("resume checkpoint encoder does not match --encoder")
        if "model" not in checkpoint or "optimizer" not in checkpoint:
            raise ValueError("resume checkpoint is not a resumable SSL checkpoint")
        model.load_state_dict(checkpoint["model"], strict=True)
        opt.load_state_dict(checkpoint["optimizer"])
        history = list(checkpoint.get("loss_history", []))
        step_history = list(checkpoint.get("step_loss_history", []))
        start_epoch = int(checkpoint.get("completed_epochs", 0))
        completed_steps = int(checkpoint.get("completed_steps", 0))
        initialization = checkpoint.get("initialization", initialization)
        print(f"resumed SSL from {args.resume} epochs={start_epoch} steps={completed_steps}")

    model.train()
    for local_epoch in range(args.epochs):
        running = 0.0
        order = list(range(len(arrays)))
        random.shuffle(order)
        for step in range(args.steps_per_epoch):
            img = arrays[order[step % len(order)]]
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
            value = float(loss.item())
            running += value
            step_history.append(value)
            completed_steps += 1
        avg = running / args.steps_per_epoch
        history.append(avg)
        absolute_epoch = start_epoch + local_epoch + 1
        print(
            f"ssl epoch={absolute_epoch} loss={avg:.6f} images={len(arrays)} "
            f"steps={completed_steps} device={device}"
        )

    completed_epochs = start_epoch + args.epochs
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 3,
        "encoder": model.encoder.state_dict(),
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "encoder_name": args.encoder,
        "input_channels": 4,
        "initialization": initialization,
        "source": "public NOAA/SWPC GONG H-alpha FITS; no external ground-truth labels",
        "source_urls": valid_urls,
        "images": len(arrays),
        "image_size": args.image_size,
        "completed_epochs": completed_epochs,
        "completed_steps": completed_steps,
        "loss_history": history,
        "step_loss_history": step_history,
        "seed": args.seed,
    }
    torch.save(payload, out)

    manifest_path = Path(args.manifest) if args.manifest else out.with_suffix(".json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {k: v for k, v in payload.items() if k not in {"encoder", "model", "optimizer"}}
    manifest["checkpoint"] = str(out)
    manifest["initial_step_loss"] = step_history[0] if step_history else None
    manifest["final_step_loss"] = step_history[-1] if step_history else None
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(out)
    print(manifest_path)


if __name__ == "__main__":
    main()
