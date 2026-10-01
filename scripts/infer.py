import argparse
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
from src.utils import image_files, load_yaml, resolve_path


ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/filanet.yaml")
ap.add_argument("--weights", nargs="+", required=True)
ap.add_argument("--output", default="submission.csv")
a = ap.parse_args()

c = load_yaml(a.config)
device = "cuda" if torch.cuda.is_available() else "cpu"
models = []

for w in a.weights:
    ck = torch.load(w, map_location=device)
    model = FILANet(
        c["model"]["encoder"],
        False,
        c["model"]["fpn_channels"],
        4,
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    models.append(model)

root = Path(c["data"]["root"])
rows = []
inf = c["inference"]

for path in image_files(resolve_path(root, c["data"]["test_images"])):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    x = build_input_channels(im)
    preds = [
        predict_tiled(m, x, device, inf["tile"], inf["overlap"], inf["tta"])
        for m in models
    ]

    mean = {
        k: sum(q[k] for q in preds) / len(preds)
        for k in ("region", "centerline", "boundary", "distance")
    }
    if "orientation" in preds[0]:
        orientation = sum(q["orientation"] for q in preds) / len(preds)
        norm = np.linalg.norm(orientation, axis=0, keepdims=True)
        mean["orientation"] = orientation / np.maximum(norm, 1e-6)

    masks = reconstruct_instances(
        mean["region"],
        mean["centerline"],
        mean["boundary"],
        mean["distance"],
        orientation=mean.get("orientation"),
        **inf,
    )
    for i, mask in enumerate(masks, 1):
        rows.append(
            {
                "filament_id": f"{path.stem}_{i}",
                "segmentation_rle": assert_roundtrip(mask),
            }
        )

pd.DataFrame(
    rows,
    columns=["filament_id", "segmentation_rle"],
).to_csv(a.output, index=False)
print("saved", a.output, len(rows))
