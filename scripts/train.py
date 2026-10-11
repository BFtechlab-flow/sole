"""Train one cross-validation fold of FILA-Net.

The checkpoint kept as ``fold{k}_best.pt`` is the one with the best OFFICIAL
score (micro Panoptic Quality, IoU > 0.5, every annotator scored separately)
on the fold's held-out observations. ``fold{k}_last.pt`` allows resuming an
interrupted run (Kaggle sessions stop after 12 h): rerun the same command.
"""

import os
import sys
from pathlib import Path as _Path

# less CUDA memory fragmentation (must be set before torch initialises CUDA)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root, so `src` imports work

import argparse
import json

from src.engine import train_fold
from src.utils import load_yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/filanet.yaml")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--output", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=None, help="override train.epochs (smoke tests)")
    ap.add_argument("--encoder-weights", default=None, help="optional GONG SSL encoder checkpoint")
    ap.add_argument("--no-resume", action="store_true", help="ignore an existing fold{k}_last.pt")
    ap.add_argument("--val-max-images", type=int, default=None, help="override train.val_max_images")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop cleanly before this wall-clock budget (Kaggle: e.g. 11); rerun to resume")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.val_max_images is not None:
        cfg["train"]["val_max_images"] = args.val_max_images
    result = train_fold(cfg, args.fold, args.output, epochs=args.epochs,
                        encoder_weights=args.encoder_weights, resume=not args.no_resume,
                        max_hours=args.max_hours)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
