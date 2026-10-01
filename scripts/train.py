import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from src.data import *
from src.inference import predict_tiled
from src.losses import total_loss
from src.metric import pq_score
from src.model import FILANet
from src.reconstruct import reconstruct_instances
from src.utils import *


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
    scores = []
    inf = config["inference"]
    threshold = config["metric"]["iou_match_threshold"]
    use_tta = bool(config["train"].get("validation_tta", False))

    for rec in records:
        image = cv2.imread(str(image_dir / rec["file_name"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_dir / rec["file_name"])
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
            **inf,
        )
        scores.append(
            pq_score(
                build_targets(rec)["instances"],
                instances,
                threshold,
            )
        )

    if not scores:
        return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "tp": 0, "fp": 0, "fn": 0}
    return {
        "pq": float(np.mean([s["pq"] for s in scores])),
        "sq": float(np.mean([s["sq"] for s in scores])),
        "rq": float(np.mean([s["rq"] for s in scores])),
        "tp": int(sum(s["tp"] for s in scores)),
        "fp": int(sum(s["fp"] for s in scores)),
        "fn": int(sum(s["fn"] for s in scores)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/filanet.yaml")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--output", default="checkpoints")
    a = ap.parse_args()

    c = load_yaml(a.config)
    seed_everything(c["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(c["data"]["root"])
    rec = load_coco_records(resolve_path(root, c["data"]["train_json"]))
    idx = np.arange(len(rec))
    groups = np.array([r["group"] for r in rec])
    tr, va = list(GroupKFold(c["n_folds"]).split(idx, groups=groups))[a.fold]
    assert not ({rec[i]["group"] for i in tr} & {rec[i]["group"] for i in va})

    image_dir = resolve_path(root, c["data"]["train_images"])
    train_ds = FilamentDataset(
        [rec[i] for i in tr],
        image_dir,
        c["train"]["patch_size"],
        True,
    )
    val_ds = FilamentDataset(
        [rec[i] for i in va],
        image_dir,
        c["train"]["patch_size"],
        False,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=c["train"]["batch_size"],
        shuffle=True,
        num_workers=c["train"]["workers"],
        pin_memory=device == "cuda",
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=c["train"].get("val_batch_size", c["train"]["batch_size"]),
        shuffle=False,
        num_workers=c["train"]["workers"],
        pin_memory=device == "cuda",
    )

    model = FILANet(
        c["model"]["encoder"],
        c["model"]["pretrained"],
        c["model"]["fpn_channels"],
        4,
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=c["train"]["lr"],
        weight_decay=c["train"]["weight_decay"],
    )
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device == "cuda" and c["train"].get("amp", True),
    )
    best_pq = -1.0
    val_records = [rec[i] for i in va]
    val_interval = int(c["train"].get("val_interval", 1))

    for epoch in range(1, c["train"]["epochs"] + 1):
        model.train()
        train_total = 0.0
        for batch in train_dl:
            x = batch["image"].to(device)
            target = {k: batch[k].to(device) for k in TARGET_KEYS}
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda",
                enabled=device == "cuda" and c["train"].get("amp", True),
            ):
                loss = total_loss(model(x), target, c["loss"])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            train_total += float(loss.item())

        train_avg = train_total / max(len(train_dl), 1)
        if epoch % val_interval != 0:
            print(f"epoch={epoch} train_loss={train_avg:.6f}")
            continue

        val_avg = validate_loss(model, val_dl, device, c["loss"])
        metrics = validate_pq(model, val_records, image_dir, device, c)
        improved = metrics["pq"] > best_pq
        print(
            f"epoch={epoch} train_loss={train_avg:.6f} val_loss={val_avg:.6f} "
            f"pq={metrics['pq']:.6f} sq={metrics['sq']:.6f} rq={metrics['rq']:.6f} "
            f"tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']} best={improved}"
        )

        if improved:
            best_pq = metrics["pq"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": c,
                    "fold": a.fold,
                    "epoch": epoch,
                    "train_loss": train_avg,
                    "validation_loss": val_avg,
                    "pq": metrics["pq"],
                    "sq": metrics["sq"],
                    "rq": metrics["rq"],
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                },
                out / f"fold{a.fold}_best.pt",
            )


if __name__ == "__main__":
    main()
