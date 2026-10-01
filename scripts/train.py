import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.cv import assert_group_isolation, fold_assignments
from src.data import FilamentDataset, TARGET_KEYS, build_input_channels, load_coco_records
from src.gt import build_instances
from src.inference import predict_tiled
from src.losses import total_loss
from src.model import FILANet
from src.pq_official import aggregate_scores, pq_score
from src.reconstruct import reconstruct_instances
from src.ssl_pretrain import load_ssl_encoder
from src.utils import load_yaml, resolve_path, seed_everything


def make_model(cfg, pretrained):
    return FILANet(
        cfg["model"]["encoder"],
        pretrained,
        cfg["model"]["fpn_channels"],
        4,
        advanced=cfg["model"].get("advanced", {}),
    )


def validate_loss(model, loader, device, loss_weights):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            target = {k: batch[k].to(device) for k in TARGET_KEYS}
            pred = model(x)
            total += float(total_loss(pred, target, loss_weights).item())
    return total / max(len(loader), 1)


def validate_pq(model, records, image_dir, device, config):
    model.eval()
    inf = config["inference"]
    threshold = float(config["metric"].get("iou_match_threshold", 0.50))
    relation_threshold = float(config["metric"].get("relation_iou_threshold", 0.10))
    use_tta = bool(config["train"].get("validation_tta", False))
    by_file = defaultdict(list)
    for record in records:
        by_file[record["file_name"]].append(record)

    scores = []
    for file_name, annotation_records in by_file.items():
        image = cv2.imread(str(image_dir / file_name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_dir / file_name)
        pred = predict_tiled(
            model,
            build_input_channels(image),
            device,
            inf["tile"],
            inf["overlap"],
            use_tta,
        )
        instances = reconstruct_instances(
            pred["region"],
            pred["centerline"],
            pred["boundary"],
            pred["distance"],
            orientation=pred.get("orientation"),
            width=pred.get("width"),
            **inf,
        )
        for record in annotation_records:
            scores.append(
                pq_score(
                    build_instances(record),
                    instances,
                    threshold=threshold,
                    relation_threshold=relation_threshold,
                )
            )

    agg = aggregate_scores(scores)
    return {
        "pq": agg["macro_pq"],
        "sq": agg["macro_sq"],
        "rq": agg["macro_rq"],
        "micro_pq": agg["micro_pq"],
        "tp": agg["tp"],
        "fp": agg["fp"],
        "fn": agg["fn"],
        "one_to_many": agg["one_to_many"],
        "many_to_one": agg["many_to_one"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/filanet.yaml")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--output", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=None, help="optional smoke/continuation override")
    ap.add_argument("--encoder-weights", default=None, help="optional GONG SSL encoder checkpoint")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(cfg["data"]["root"])
    records = load_coco_records(resolve_path(root, cfg["data"]["train_json"]))
    mode = cfg.get("cv", {}).get("group_mode", "observation")
    folds = fold_assignments(records, cfg["n_folds"], mode)
    assert_group_isolation(records, folds, mode)
    tr = np.where(folds != args.fold)[0]
    va = np.where(folds == args.fold)[0]
    image_dir = resolve_path(root, cfg["data"]["train_images"])
    width_scale = float(cfg.get("targets", {}).get("width_scale_px", 64.0))

    train_ds = FilamentDataset(
        [records[i] for i in tr], image_dir, cfg["train"]["patch_size"], True,
        width_scale_px=width_scale,
    )
    val_ds = FilamentDataset(
        [records[i] for i in va], image_dir, cfg["train"]["patch_size"], False,
        width_scale_px=width_scale,
    )
    train_dl = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["train"]["workers"], pin_memory=device == "cuda",
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg["train"].get("val_batch_size", cfg["train"]["batch_size"]),
        shuffle=False, num_workers=cfg["train"]["workers"], pin_memory=device == "cuda",
    )

    model = make_model(cfg, cfg["model"]["pretrained"]).to(device)
    if args.encoder_weights:
        ssl = load_ssl_encoder(model, args.encoder_weights, strict=True)
        if ssl.get("encoder_name") and ssl["encoder_name"] != cfg["model"]["encoder"]:
            raise ValueError(
                f"SSL encoder name {ssl['encoder_name']} does not match configured {cfg['model']['encoder']}"
            )
        print(
            f"loaded SSL encoder from {args.encoder_weights}; "
            f"images={ssl.get('images', 'unknown')} steps={ssl.get('completed_steps', 'unknown')}"
        )
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    epochs = int(args.epochs if args.epochs is not None else cfg["train"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(epochs, 1), eta_min=float(cfg["train"].get("min_lr", 1e-6))
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device == "cuda" and cfg["train"].get("amp", True)
    )
    best_pq = -1.0
    val_records = [records[i] for i in va]
    val_interval = int(cfg["train"].get("val_interval", 1))
    grad_clip = float(cfg["train"].get("grad_clip_norm", 1.0))

    print(
        f"training fold={args.fold} device={device} epochs={epochs} "
        f"advanced={bool(cfg['model'].get('advanced', {}).get('enabled', False))}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        for batch in train_dl:
            x = batch["image"].to(device)
            target = {k: batch[k].to(device) for k in TARGET_KEYS}
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda", enabled=device == "cuda" and cfg["train"].get("amp", True)
            ):
                loss = total_loss(model(x), target, cfg["loss"])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            train_total += float(loss.item())
        scheduler.step()

        train_avg = train_total / max(len(train_dl), 1)
        if epoch % val_interval != 0:
            print(f"epoch={epoch} train_loss={train_avg:.6f} lr={scheduler.get_last_lr()[0]:.3e}")
            continue

        val_avg = validate_loss(model, val_dl, device, cfg["loss"])
        metrics = validate_pq(model, val_records, image_dir, device, cfg)
        improved = metrics["pq"] > best_pq
        print(
            f"epoch={epoch} train_loss={train_avg:.6f} val_loss={val_avg:.6f} "
            f"macro_pq={metrics['pq']:.6f} micro_pq={metrics['micro_pq']:.6f} "
            f"sq={metrics['sq']:.6f} rq={metrics['rq']:.6f} tp={metrics['tp']} "
            f"fp={metrics['fp']} fn={metrics['fn']} lr={scheduler.get_last_lr()[0]:.3e} "
            f"best={improved}"
        )
        if improved:
            best_pq = metrics["pq"]
            torch.save(
                {
                    "schema_version": 2,
                    "model": model.state_dict(), "config": cfg, "fold": args.fold,
                    "group_mode": mode, "epoch": epoch, "train_loss": train_avg,
                    "validation_loss": val_avg, "pq": metrics["pq"],
                    "micro_pq": metrics["micro_pq"], "sq": metrics["sq"],
                    "rq": metrics["rq"], "tp": metrics["tp"], "fp": metrics["fp"],
                    "fn": metrics["fn"],
                },
                out / f"fold{args.fold}_best.pt",
            )


if __name__ == "__main__":
    main()
