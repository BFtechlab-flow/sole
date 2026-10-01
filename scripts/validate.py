import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from src.data import build_input_channels, build_targets, load_coco_records
from src.inference import predict_tiled
from src.metric import pq_score
from src.model import FILANet
from src.reconstruct import reconstruct_instances
from src.utils import load_yaml, resolve_path


ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/filanet.yaml")
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--weights", required=True)
a = ap.parse_args()

c = load_yaml(a.config)
device = "cuda" if torch.cuda.is_available() else "cpu"
root = Path(c["data"]["root"])
records = load_coco_records(resolve_path(root, c["data"]["train_json"]))
idx = np.arange(len(records))
groups = np.array([x["group"] for x in records])
_, va = list(GroupKFold(c["n_folds"]).split(idx, groups=groups))[a.fold]

ck = torch.load(a.weights, map_location=device)
model = FILANet(
    c["model"]["encoder"],
    False,
    c["model"]["fpn_channels"],
    4,
).to(device)
model.load_state_dict(ck["model"])
model.eval()

scores = []
imgdir = resolve_path(root, c["data"]["train_images"])
inf = c["inference"]

for i in va:
    rec = records[i]
    im = cv2.imread(str(imgdir / rec["file_name"]), cv2.IMREAD_GRAYSCALE)
    pred = predict_tiled(
        model,
        build_input_channels(im),
        device,
        inf["tile"],
        inf["overlap"],
        inf["tta"],
    )
    pm = reconstruct_instances(
        pred["region"],
        pred["centerline"],
        pred["boundary"],
        pred["distance"],
        orientation=pred.get("orientation"),
        **inf,
    )
    gt = build_targets(rec)["instances"]
    scores.append(pq_score(gt, pm, c["metric"]["iou_match_threshold"]))

print(
    json.dumps(
        {
            k: float(np.mean([x[k] for x in scores]))
            for k in ("pq", "sq", "rq", "tp", "fp", "fn")
        },
        indent=2,
    )
)
