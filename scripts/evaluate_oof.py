import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.cv import assert_group_isolation, fold_assignments
from src.data import load_coco_records
from src.gt import build_instances
from src.oof import cache_path, load_prediction
from src.pq_official import aggregate_scores, bootstrap_macro_pq, pq_score
from src.reconstruct import reconstruct_instances
from src.utils import load_yaml, resolve_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/filanet.yaml")
    ap.add_argument("--cache-dir", default="artifacts/oof_cache")
    ap.add_argument("--output", default="artifacts/oof_metrics.json")
    ap.add_argument("--records-csv", default="artifacts/oof_records.csv")
    ap.add_argument("--bootstrap", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    root = Path(cfg["data"]["root"])
    records = load_coco_records(resolve_path(root, cfg["data"]["train_json"]))
    mode = cfg.get("cv", {}).get("group_mode", "observation")
    folds = fold_assignments(records, cfg["n_folds"], mode)
    assert_group_isolation(records, folds, mode)
    threshold = float(cfg["metric"].get("iou_match_threshold", 0.5))
    relation_threshold = float(cfg["metric"].get("relation_iou_threshold", 0.10))
    inf = cfg["inference"]

    by_fold_file = defaultdict(list)
    for idx, (record, fold) in enumerate(zip(records, folds)):
        by_fold_file[(int(fold), record["file_name"])].append(idx)

    scores = []
    rows = []
    total_images = len(by_fold_file)
    for image_no, ((fold, file_name), record_indices) in enumerate(by_fold_file.items(), 1):
        path = cache_path(args.cache_dir, fold, file_name)
        if not path.exists():
            raise FileNotFoundError(f"missing OOF cache {path}; run scripts/cache_oof.py first")
        pred = load_prediction(path)
        predicted_instances = reconstruct_instances(
            pred["region"], pred["centerline"], pred["boundary"], pred["distance"],
            orientation=pred.get("orientation"), **inf,
        )
        for idx in record_indices:
            record = records[idx]
            score = pq_score(
                build_instances(record), predicted_instances,
                threshold=threshold, relation_threshold=relation_threshold,
            )
            scores.append(score)
            rows.append({
                "record_index": idx,
                "image_id": record["image_id"],
                "file_name": record["file_name"],
                "fold": fold,
                "pq": score["pq"],
                "sq": score["sq"],
                "rq": score["rq"],
                "tp": score["tp"],
                "fp": score["fp"],
                "fn": score["fn"],
                "one_to_many": score["one_to_many"],
                "many_to_one": score["many_to_one"],
                "gt_count": score["gt_count"],
                "pred_count": score["pred_count"],
            })
        if image_no % 25 == 0 or image_no == total_images:
            print(f"evaluated {image_no}/{total_images} unique OOF images")
        del pred, predicted_instances

    aggregate = aggregate_scores(scores)
    iterations = int(args.bootstrap) if args.bootstrap is not None else int(cfg["metric"].get("bootstrap_iterations", 1000))
    aggregate["bootstrap_macro_pq"] = bootstrap_macro_pq(scores, iterations=iterations, seed=cfg["seed"])
    aggregate["fold_macro_pq"] = {
        str(f): float(np.mean([r["pq"] for r in rows if r["fold"] == f]))
        for f in range(cfg["n_folds"])
    }
    aggregate["group_mode"] = mode
    aggregate["iou_match_threshold"] = threshold
    aggregate["relation_iou_threshold"] = relation_threshold

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    csv_path = Path(args.records_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
