import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import torch

from src.cv import assert_group_isolation, fold_assignments
from src.data import build_input_channels, load_coco_records
from src.gt import build_instances
from src.inference import predict_tiled
from src.model import FILANet
from src.pq_official import aggregate_scores, bootstrap_macro_pq, pq_score
from src.reconstruct import reconstruct_instances
from src.utils import load_yaml, resolve_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/filanet.yaml")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(cfg["data"]["root"])
    records = load_coco_records(resolve_path(root, cfg["data"]["train_json"]))
    mode = cfg.get("cv", {}).get("group_mode", "observation")
    folds = fold_assignments(records, cfg["n_folds"], mode)
    assert_group_isolation(records, folds, mode)
    val_records = [r for r, f in zip(records, folds) if int(f) == args.fold]
    ck = torch.load(args.weights, map_location=device)
    model = FILANet(cfg["model"]["encoder"], False, cfg["model"]["fpn_channels"], 4).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    image_dir = resolve_path(root, cfg["data"]["train_images"])
    inf = cfg["inference"]
    threshold = float(cfg["metric"].get("iou_match_threshold", 0.50))
    relation_threshold = float(cfg["metric"].get("relation_iou_threshold", 0.10))
    by_file = defaultdict(list)
    for record in val_records:
        by_file[record["file_name"]].append(record)
    scores = []
    for pos, (file_name, annotation_records) in enumerate(by_file.items(), 1):
        image = cv2.imread(str(image_dir / file_name), cv2.IMREAD_GRAYSCALE)
        if image is None: raise FileNotFoundError(image_dir / file_name)
        pred = predict_tiled(model, build_input_channels(image), device, inf["tile"], inf["overlap"], inf["tta"])
        instances = reconstruct_instances(pred["region"], pred["centerline"], pred["boundary"], pred["distance"], orientation=pred.get("orientation"), **inf)
        for record in annotation_records:
            scores.append(pq_score(build_instances(record), instances, threshold=threshold, relation_threshold=relation_threshold))
        if pos % 10 == 0 or pos == len(by_file): print(f"validated {pos}/{len(by_file)} unique images")
    result = aggregate_scores(scores)
    result["bootstrap_macro_pq"] = bootstrap_macro_pq(scores, iterations=int(cfg["metric"].get("bootstrap_iterations", 1000)), seed=cfg["seed"] + args.fold)
    result["fold"] = args.fold; result["group_mode"] = mode; result["checkpoint"] = args.weights
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
