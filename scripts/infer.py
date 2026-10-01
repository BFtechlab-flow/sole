import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from src.data import build_input_channels
from src.inference import predict_tiled
from src.model import FILANet
from src.reconstruct import reconstruct_instances
from src.rle import assert_roundtrip
from src.submission import save_verification, validate_submission_frame
from src.utils import image_files, load_yaml, resolve_path


def build_model(cfg, device):
    return FILANet(
        cfg["model"]["encoder"],
        False,
        cfg["model"]["fpn_channels"],
        4,
        advanced=cfg["model"].get("advanced", {}),
    ).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/filanet.yaml")
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--output", default="submission.csv")
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for weights in args.weights:
        ck = torch.load(weights, map_location=device)
        model = build_model(cfg, device)
        model.load_state_dict(ck["model"])
        model.eval()
        models.append(model)

    root = Path(cfg["data"]["root"])
    test_dir = resolve_path(root, cfg["data"]["test_images"])
    rows = []
    image_shapes = {}
    inf = cfg["inference"]

    paths = image_files(test_dir)
    for pos, path in enumerate(paths, 1):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        image_shapes[path.stem] = image.shape
        x = build_input_channels(image)
        preds = [
            predict_tiled(m, x, device, inf["tile"], inf["overlap"], inf["tta"])
            for m in models
        ]

        keys = ["region", "centerline", "boundary", "distance"]
        for optional in ("width", "curvature", "endpoint", "junction", "uncertainty"):
            if all(optional in p for p in preds):
                keys.append(optional)
        mean = {key: sum(q[key] for q in preds) / len(preds) for key in keys}

        if "orientation" in preds[0]:
            orientation = sum(q["orientation"] for q in preds) / len(preds)
            norm = np.linalg.norm(orientation, axis=0, keepdims=True)
            mean["orientation"] = orientation / np.maximum(norm, 1e-6)

        masks = reconstruct_instances(
            mean["region"], mean["centerline"], mean["boundary"], mean["distance"],
            orientation=mean.get("orientation"), width=mean.get("width"), **inf,
        )
        for i, mask in enumerate(masks, 1):
            rows.append({"filament_id": f"{path.stem}_{i}", "segmentation_rle": assert_roundtrip(mask)})
        if pos % 10 == 0 or pos == len(paths):
            print(f"inference {pos}/{len(paths)} images; rows={len(rows)}")

    frame = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    frame.to_csv(args.output, index=False)
    print("saved", args.output, len(rows))

    if not args.skip_validation:
        report = validate_submission_frame(frame, image_shapes, require_nonoverlap=True)
        verification = str(Path(args.output).with_suffix(".verification.json"))
        report = save_verification(report, args.output, verification)
        print(json.dumps(report, indent=2))
        if not report["ok"]:
            raise SystemExit("submission validation failed")


if __name__ == "__main__":
    main()
