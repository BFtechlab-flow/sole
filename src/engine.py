"""Training / validation engine shared by the scripts and the notebook."""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .cv import assert_group_isolation, fold_assignments
from .data import TARGET_KEYS, FilamentDataset, build_input_channels, load_coco_records
from .evaluation import group_records_by_image, label_map_to_rles, score_image, subsample_images, summarize
from .inference import ensemble_predict
from .losses import total_loss
from .model import FILANet
from .oof import quantize_roundtrip
from .reconstruct import reconstruct_label_map
from .ssl_pretrain import load_ssl_encoder
from .utils import data_paths, seed_everything

CHECKPOINT_SCHEMA = 4


# --------------------------------------------------------------------------- models
def make_model(cfg, pretrained: bool):
    m = cfg["model"]
    return FILANet(
        m["encoder"],
        pretrained,
        m["fpn_channels"],
        4,
        advanced=m.get("advanced", {}),
        full_res_refine=bool(m.get("full_res_refine", False)),
    )


def load_models(cfg, weights, device):
    """Load fold checkpoints for inference (architecture from ``cfg``)."""
    models = []
    for w in weights:
        ck = torch.load(w, map_location=device, weights_only=True)
        model = make_model(cfg, pretrained=False).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        models.append(model)
    return models


#: keys of the ``inference`` section that control the network forward pass
#: (everything else parameterises the instance reconstruction).
PREDICTION_KEYS = ("tile", "overlap", "tta", "amp")


def postprocess_params(cfg) -> dict:
    """Reconstruction parameters = the ``inference`` section minus tiling/TTA/AMP keys."""
    return {k: v for k, v in cfg["inference"].items() if k not in PREDICTION_KEYS}


def predict_label_map(models, gray, cfg, device, tta=None):
    """Full inference for one grayscale image: tiled ensemble -> quantised maps -> instances."""
    inf = cfg["inference"]
    use_tta = inf.get("tta", True) if tta is None else tta
    pred = ensemble_predict(models, build_input_channels(gray), device, inf["tile"], inf["overlap"], use_tta,
                            amp=bool(inf.get("amp", False)))
    pred = quantize_roundtrip(pred)
    labels, scores = reconstruct_label_map(
        pred["region"], pred["centerline"], pred["boundary"], pred["distance"],
        orientation=pred.get("orientation"), width=pred.get("width"),
        instance_embedding=pred.get("instance_embedding"), **postprocess_params(cfg),
    )
    return labels, scores, pred


# --------------------------------------------------------------------------- folds
def load_records_and_folds(cfg):
    root, train_json, train_images, _ = data_paths(cfg)
    records = load_coco_records(train_json)
    mode = cfg.get("cv", {}).get("group_mode", "observation")
    folds = fold_assignments(records, cfg["n_folds"], mode)
    assert_group_isolation(records, folds, mode)
    return records, folds, train_images, mode


# --------------------------------------------------------------------------- validation
@torch.no_grad()
def validate_pq(model, records, indices, image_dir, device, cfg, max_images=None, tta=False):
    """Official micro PQ of ``model`` on the validation images (all annotators scored)."""
    model.eval()
    metric = cfg.get("metric", {})
    threshold = float(metric.get("iou_match_threshold", 0.5))
    rel = float(metric.get("relation_iou_threshold", 0.10))
    by_file = group_records_by_image(records, indices)
    files = subsample_images(list(by_file), max_images, seed=int(cfg.get("seed", 2026)))
    scores = []
    for file_name in files:
        gray = cv2.imread(str(Path(image_dir) / file_name), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(Path(image_dir) / file_name)
        labels, _, _ = predict_label_map([model], gray, cfg, device, tta=tta)
        scores += score_image([records[i] for i in by_file[file_name]], label_map_to_rles(labels), threshold, rel)
    out = summarize(scores)
    out["validation_images"] = len(files)
    return out


@torch.no_grad()
def validate_loss(model, loader, device, loss_weights):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        x = batch["image"].to(device)
        target = {k: batch[k].to(device) for k in TARGET_KEYS}
        total += float(total_loss(model(x), target, loss_weights).item())
        n += 1
    return total / max(n, 1)


# --------------------------------------------------------------------------- training
class ModelEMA:
    """Exponential moving average of the weights (optional, ``train.ema_decay``)."""

    def __init__(self, model, decay):
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])


def _save(path, payload):
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def train_fold(cfg, fold, output, epochs=None, encoder_weights=None, resume="auto", log=print,
               max_hours=None):
    """Train one fold; keep the checkpoint with the best official (micro) PQ.

    Writes ``fold{k}_best.pt``, ``fold{k}_last.pt`` (resumable) and
    ``fold{k}_log.jsonl`` (one line per epoch) into ``output``.

    ``max_hours`` (or ``train.max_hours``): stop cleanly after an epoch when
    the next epoch (training + validation, estimated from the slowest epoch so
    far) would exceed the budget. Kaggle sessions end after 12 h; rerunning
    the same command (with ``fold{k}_last.pt`` in ``output``) resumes.
    """
    t_start = time.time()
    if max_hours is None:
        max_hours = cfg.get("train", {}).get("max_hours")
    seed_everything(int(cfg["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records, folds, image_dir, mode = load_records_and_folds(cfg)
    tr = np.where(folds != fold)[0]
    va = np.where(folds == fold)[0]
    tcfg = cfg["train"]
    width_scale = float(cfg.get("targets", {}).get("width_scale_px", 64.0))
    workers = int(tcfg.get("workers", 4))

    train_ds = FilamentDataset([records[i] for i in tr], image_dir, tcfg["patch_size"], True, width_scale_px=width_scale)
    val_ds = FilamentDataset([records[i] for i in va], image_dir, tcfg["patch_size"], False, width_scale_px=width_scale)
    train_dl = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, num_workers=workers,
                          pin_memory=device == "cuda", persistent_workers=workers > 0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=tcfg.get("val_batch_size", tcfg["batch_size"]), shuffle=False,
                        num_workers=workers, pin_memory=device == "cuda", persistent_workers=workers > 0)

    model = make_model(cfg, bool(cfg["model"].get("pretrained", True))).to(device)
    if encoder_weights:
        ssl = load_ssl_encoder(model, encoder_weights, strict=True)
        if ssl.get("encoder_name") and ssl["encoder_name"] != cfg["model"]["encoder"]:
            raise ValueError(f"SSL encoder {ssl['encoder_name']} does not match {cfg['model']['encoder']}")
        log(f"loaded SSL encoder {encoder_weights} (images={ssl.get('images')}, steps={ssl.get('completed_steps')})")

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    n_epochs = int(epochs if epochs is not None else tcfg["epochs"])
    steps_per_epoch = max(len(train_dl), 1)
    warmup = int(tcfg.get("warmup_epochs", 0) * steps_per_epoch)
    total_steps = n_epochs * steps_per_epoch
    min_ratio = float(tcfg.get("min_lr", 1e-6)) / float(tcfg["lr"])

    def lr_lambda(step):  # linear warm-up then cosine (per optimisation step)
        if warmup and step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(total_steps - warmup, 1)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    amp = device == "cuda" and bool(tcfg.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = ModelEMA(model, tcfg["ema_decay"]) if float(tcfg.get("ema_decay", 0) or 0) > 0 else None

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    best_path, last_path, log_path = out / f"fold{fold}_best.pt", out / f"fold{fold}_last.pt", out / f"fold{fold}_log.jsonl"
    start_epoch, best_pq = 1, -1.0
    if resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_raw"])
        opt.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        if ema is not None and ck.get("ema") is not None:
            ema.module.load_state_dict(ck["ema"])
        start_epoch, best_pq = int(ck["epoch"]) + 1, float(ck["best_pq"])
        log(f"resumed fold {fold} from epoch {ck['epoch']} (best micro PQ {best_pq:.4f})")

    val_interval = int(tcfg.get("val_interval", 1))
    val_max_images = tcfg.get("val_max_images")
    grad_clip = float(tcfg.get("grad_clip_norm", 1.0))
    log(f"fold={fold} device={device} epochs={n_epochs} train_records={len(tr)} val_records={len(va)} group_mode={mode}")
    slowest_train, slowest_val = 0.0, None  # longest training / validation pass so far (time budget)

    for epoch in range(start_epoch, n_epochs + 1):
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for batch in train_dl:
            x = batch["image"].to(device, non_blocking=True)
            target = {k: batch[k].to(device, non_blocking=True) for k in TARGET_KEYS}
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp):
                loss = total_loss(model(x), target, cfg["loss"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            run += float(loss.item())
            n += 1
        row = {"fold": fold, "epoch": epoch, "train_loss": run / max(n, 1), "lr": scheduler.get_last_lr()[0],
               "epoch_seconds": round(time.time() - t0, 1)}
        eval_model = ema.module if ema is not None else model
        if epoch % val_interval == 0 or epoch == n_epochs:
            row["val_loss"] = validate_loss(eval_model, val_dl, device, cfg["loss"])
            m = validate_pq(eval_model, records, va, image_dir, device, cfg, max_images=val_max_images,
                            tta=bool(tcfg.get("validation_tta", False)))
            row.update({k: m[k] for k in ("pq", "sq", "rq", "tp", "fp", "fn", "macro_pq_diagnostic",
                                          "fragmentation_rate", "merge_rate", "validation_images")})
            row["best"] = bool(m["pq"] > best_pq)
            if row["best"]:
                best_pq = float(m["pq"])
                _save(best_path, {"schema_version": CHECKPOINT_SCHEMA, "model": eval_model.state_dict(), "config": cfg,
                                  "fold": int(fold), "group_mode": mode, "epoch": int(epoch),
                                  "metrics": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in row.items()}})
        _save(last_path, {"schema_version": CHECKPOINT_SCHEMA, "model_raw": model.state_dict(),
                          "ema": ema.module.state_dict() if ema is not None else None,
                          "optimizer": opt.state_dict(), "scheduler": scheduler.state_dict(),
                          "scaler": scaler.state_dict(), "epoch": int(epoch), "best_pq": float(best_pq), "config": cfg})
        row["wall_seconds"] = round(time.time() - t0, 1)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        log(" ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        train_s = float(row["epoch_seconds"])
        val_s = float(row["wall_seconds"]) - train_s
        slowest_train = max(slowest_train, train_s)
        if "pq" in row:
            slowest_val = max(slowest_val or 0.0, val_s)
        if max_hours and epoch < n_epochs:
            elapsed = time.time() - t_start
            nxt = epoch + 1
            validates = nxt % val_interval == 0 or nxt == n_epochs
            # unknown validation cost: assume it is as long as a training epoch
            estimate = 1.1 * (slowest_train + ((slowest_val if slowest_val is not None else slowest_train)
                                               if validates else 0.0))
            if elapsed + estimate > float(max_hours) * 3600.0:
                log(f"time budget: stopping after epoch {epoch} ({elapsed / 3600:.2f} h used of {float(max_hours):.2f} h); "
                    f"rerun the same command to resume from {last_path.name}")
                return {"fold": int(fold), "best_pq": float(best_pq), "best_checkpoint": str(best_path),
                        "completed_epochs": int(epoch), "epochs": n_epochs, "finished": False}
    return {"fold": int(fold), "best_pq": float(best_pq), "best_checkpoint": str(best_path),
            "completed_epochs": int(n_epochs), "epochs": n_epochs, "finished": True}
