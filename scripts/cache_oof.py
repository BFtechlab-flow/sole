"""Cache out-of-fold predictions: every training image is predicted by the
model of the fold in which it was held out (once per physical image, even
when 2-3 annotators labelled it). Maps are stored in the compact quantised
format of ``src/oof.py``.
"""

import os
import sys
from pathlib import Path as _Path

# less CUDA memory fragmentation (must be set before torch initialises CUDA)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root, so `src` imports work

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.data import build_input_channels
from src.engine import load_models, load_records_and_folds
from src.inference import ensemble_predict
from src.oof import available_prediction, cache_path, save_prediction
from src.utils import load_yaml, seed_everything


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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/filanet.yaml")
    ap.add_argument("--weights-pattern", nargs="+", default=["checkpoints/fold{fold}_best.pt"],
                    help="one pattern per model; several patterns = per-fold ensemble (maps averaged as at test time)")
    ap.add_argument("--cache-dir", default="artifacts/oof_cache")
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--tta", choices=["config", "on", "off"], default="config")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop cleanly before this budget (cached files are kept; rerun to resume)")
    args = ap.parse_args()
    deadline = time.time() + 3600.0 * args.max_hours if args.max_hours else None

    cfg = load_yaml(args.config)
    seed_everything(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records, folds, image_dir, mode = load_records_and_folds(cfg)
    selected = args.folds if args.folds else list(range(cfg["n_folds"]))
    inf = cfg["inference"]
    use_tta = resolve_tta(args, cfg)

    manifest = {"config": args.config, "group_mode": mode, "tta": use_tta, "amp": bool(inf.get("amp", False)),
                "device": device, "folds": {}}
    manifest_path = Path(args.manifest) if args.manifest else Path(args.cache_dir) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def write_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for fold in selected:
        weights = [Path(p.format(fold=fold)) for p in args.weights_pattern]
        for w in weights:
            if not w.exists():
                raise FileNotFoundError(w)
        shas = [sha256_file(w) for w in weights]
        # recorded before any file is written, so that partial caches can be traced to their weights
        manifest["folds"][str(fold)] = {"weights": [str(w) for w in weights] if len(weights) > 1 else str(weights[0]),
                                        "weights_sha256": "+".join(shas), "complete": False}
        write_manifest()
        models = load_models(cfg, weights, device)
        files = sorted({records[i]["file_name"] for i in np.where(folds == fold)[0]})
        written, t0 = 0, time.time()
        for pos, file_name in enumerate(files, 1):
            out_path = cache_path(args.cache_dir, fold, file_name)
            if not args.overwrite and available_prediction(out_path):
                continue
            if deadline is not None and time.time() > deadline:
                print(f"time budget used (fold {fold}, {pos - 1}/{len(files)} images): rerun to continue", flush=True)
                raise SystemExit(0)
            image = cv2.imread(str(Path(image_dir) / file_name), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(Path(image_dir) / file_name)
            pred = ensemble_predict(models, build_input_channels(image), device, inf["tile"], inf["overlap"], use_tta,
                                    amp=bool(inf.get("amp", False)))
            save_prediction(out_path, pred)
            written += 1
            if pos % 10 == 0 or pos == len(files):
                print(f"fold={fold} cached={pos}/{len(files)}", flush=True)
        manifest["folds"][str(fold)].update({
            "complete": True, "validation_images": len(files), "new_cache_files": written,
            "seconds": round(time.time() - t0, 1),
        })
        write_manifest()
        del models
        if device == "cuda":
            torch.cuda.empty_cache()

    write_manifest()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
