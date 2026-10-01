import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.cv import assert_group_isolation, fold_assignments
from src.data import build_input_channels, load_coco_records
from src.inference import predict_tiled
from src.model import FILANet
from src.oof import available_prediction, cache_path, save_prediction
from src.utils import load_yaml, resolve_path, seed_everything


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_tta(args, cfg):
    if args.tta == "on":
        return True
    if args.tta == "off":
        return False
    return bool(cfg["inference"].get("tta", True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/filanet.yaml")
    ap.add_argument("--weights-pattern", default="checkpoints/fold{fold}_best.pt")
    ap.add_argument("--cache-dir", default="artifacts/oof_cache")
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--tta", choices=["config", "on", "off"], default="config")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--manifest", default="artifacts/oof_cache/manifest.json")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(cfg["data"]["root"])
    records = load_coco_records(resolve_path(root, cfg["data"]["train_json"]))
    mode = cfg.get("cv", {}).get("group_mode", "observation")
    folds = fold_assignments(records, cfg["n_folds"], mode)
    assert_group_isolation(records, folds, mode)
    image_dir = resolve_path(root, cfg["data"]["train_images"])
    selected = args.folds or list(range(cfg["n_folds"]))
    inf = cfg["inference"]
    use_tta = resolve_tta(args, cfg)

    manifest = {
        "config": args.config,
        "group_mode": mode,
        "tta": use_tta,
        "device": device,
        "folds": {},
    }

    for fold in selected:
        weights = Path(args.weights_pattern.format(fold=fold))
        if not weights.exists():
            raise FileNotFoundError(weights)
        checkpoint = torch.load(weights, map_location=device)
        model = FILANet(cfg["model"]["encoder"], False, cfg["model"]["fpn_channels"], 4).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        files = sorted({records[i]["file_name"] for i in np.where(folds == fold)[0]})
        written = 0
        for pos, file_name in enumerate(files, 1):
            out_path = cache_path(args.cache_dir, fold, file_name)
            if not args.overwrite and available_prediction(out_path):
                continue
            image = cv2.imread(str(image_dir / file_name), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(image_dir / file_name)
            pred = predict_tiled(model, build_input_channels(image), device, inf["tile"], inf["overlap"], use_tta)
            save_prediction(out_path, pred)
            written += 1
            if pos % 10 == 0 or pos == len(files):
                print(f"fold={fold} cached={pos}/{len(files)}")

        manifest["folds"][str(fold)] = {
            "weights": str(weights),
            "weights_sha256": sha256_file(weights),
            "validation_images": len(files),
            "new_cache_files": written,
        }
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
