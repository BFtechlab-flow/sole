"""Official-metric validation of one fold checkpoint on its held-out observations."""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root, so `src` imports work

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.engine import load_models, load_records_and_folds, validate_pq
from src.utils import load_yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/filanet.yaml")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records, folds, image_dir, mode = load_records_and_folds(cfg)
    va = np.where(folds == args.fold)[0]
    model = load_models(cfg, [args.weights], device)[0]
    result = validate_pq(model, records, va, image_dir, device, cfg, max_images=args.max_images, tta=args.tta)
    result.update({"fold": args.fold, "group_mode": mode, "checkpoint": args.weights})
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("distribution")}, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
