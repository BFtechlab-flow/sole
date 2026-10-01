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
from src.utils import load_yaml, resolve_path, seed_everything


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
        pred = predict_tiled(model, build_input_channels(image), device, inf["tile"], inf["overlap"], use_tta)
        instances = reconstruct_instances(
            pred["region"], pred["centerline"], pred["boundary"], pred["distance"],
            orientation=pred.get("orientation"), **inf,
        )
        for record in annotation_records:
            scores.append(pq_score(
                build_instances(record), instances,
                threshold=threshold, relation_threshold=relation_threshold,
            ))

    agg = aggregate_scores(scores)
    return {
        "pq": agg["macro_pq"], "sq": agg["macro_sq"], "rq": agg["macro_rq"],
        "micro_pq": agg["micro_pq"], "tp": agg["tp"], "fp": agg["fp"], "fn": agg["fn"],
        "one_to_many": agg["one_to_many"], "many_to_one": agg["many_to_one"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/filanet.yaml")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--output", default="checkpoints")
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
    train_ds = FilamentDataset([records[i] for i in tr], image_dir, cfg["train"]["patch_size"], True)
    val_ds = FilamentDataset([records[i] for i in va], image_dir, cfg["train"]["patch_size"], False)
    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["workers"], pin_memory=device == "cuda")
    val_dl = DataLoader(val_ds, batch_size=cfg["train"].get("val_batch_size", cfg["train"]["batch_size"]), shuffle=False, num_workers=cfg["train"]["workers"], pin_memory=device == "cuda")
    model = FILANet(cfg["model"]["encoder"], cfg["model"]["pretrained"], cfg["model"]["fpn_channels"], 4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and cfg["train"].get("amp", True))
    best_pq = -1.0
    val_records = [records[i] for i in va]
    val_interval = int(cfg["train"].get("val_interval", 1))

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train(); train_total = 0.0
        for batch in train_dl:
            x = batch["image"].to(device)
            target = {k: batch[k].to(device) for k in TARGET_KEYS}
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device == "cuda" and cfg["train"].get("amp", True)):
                loss = total_loss(model(x), target, cfg["loss"])
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            train_total += float(loss.item())
        train_avg = train_total / max(len(train_dl), 1)
        if epoch % val_interval != 0:
            print(f"epoch={epoch} train_loss={train_avg:.6f}"); continue
        val_avg = validate_loss(model, val_dl, device, cfg["loss"])
        metrics = validate_pq(model, val_records, image_dir, device, cfg)
        improved = metrics["pq"] > best_pq
        print(f"epoch={epoch} train_loss={train_avg:.6f} val_loss={val_avg:.6f} macro_pq={metrics['pq']:.6f} micro_pq={metrics['micro_pq']:.6f} sq={metrics['sq']:.6f} rq={metrics['rq']:.6f} tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']} best={improved}")
        if improved:
            best_pq = metrics["pq"]
            torch.save({
                "model": model.state_dict(), "config": cfg, "fold": args.fold, "group_mode": mode,
                "epoch": epoch, "train_loss": train_avg, "validation_loss": val_avg,
                "pq": metrics["pq"], "micro_pq": metrics["micro_pq"], "sq": metrics["sq"], "rq": metrics["rq"],
                "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"],
            }, out / f"fold{args.fold}_best.pt")


if __name__ == "__main__":
    main()
