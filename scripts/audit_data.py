import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.data import build_targets, load_coco_records
from src.cv import fold_assignments, assert_group_isolation
from src.utils import load_yaml, resolve_path


EXPECTED = {
    "unique_observations": 707,
    "annotation_records": 1154,
    "filament_instances": 8199,
    "image_shape": (2048, 2048),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/filanet.yaml")
    ap.add_argument("--decode-all", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    root = Path(cfg["data"]["root"])
    json_path = resolve_path(root, cfg["data"]["train_json"])
    image_dir = resolve_path(root, cfg["data"]["train_images"])
    records = load_coco_records(json_path)

    groups = Counter(r["group"] for r in records)
    instance_count = int(sum(len(r["annotations"]) for r in records))
    sizes = sorted(set((r["height"], r["width"]) for r in records))
    missing = sorted({r["file_name"] for r in records if not (image_dir / r["file_name"]).exists()})

    mode = cfg.get("cv", {}).get("group_mode", "observation")
    folds = fold_assignments(records, cfg["n_folds"], mode)
    assert_group_isolation(records, folds, mode)

    report = {
        "annotation_records": len(records),
        "unique_observations": len(groups),
        "repeated_observations": int(sum(v > 1 for v in groups.values())),
        "filament_instances": instance_count,
        "sizes": sizes,
        "mean_instances_per_record": float(np.mean([len(r["annotations"]) for r in records])),
        "missing_images": missing,
        "fold_record_counts": {str(f): int(np.sum(folds == f)) for f in range(cfg["n_folds"])},
        "group_mode": mode,
    }

    if args.decode_all:
        areas = []
        for i, record in enumerate(records):
            targets = build_targets(record)
            areas.extend(int(m.sum()) for m in targets["instances"])
            if i % 100 == 0:
                print(f"decoded {i}/{len(records)}")
        report["decoded_instances"] = len(areas)
        report["instance_area"] = {
            "min": int(min(areas)) if areas else None,
            "median": float(np.median(areas)) if areas else None,
            "max": int(max(areas)) if areas else None,
        }

    mismatches = []
    if len(records) != EXPECTED["annotation_records"]:
        mismatches.append(f"annotation_records={len(records)} expected={EXPECTED['annotation_records']}")
    if len(groups) != EXPECTED["unique_observations"]:
        mismatches.append(f"unique_observations={len(groups)} expected={EXPECTED['unique_observations']}")
    if instance_count != EXPECTED["filament_instances"]:
        mismatches.append(f"filament_instances={instance_count} expected={EXPECTED['filament_instances']}")
    if sizes != [EXPECTED["image_shape"]]:
        mismatches.append(f"sizes={sizes} expected={[EXPECTED['image_shape']]}")
    if missing:
        mismatches.append(f"{len(missing)} referenced train images are missing")
    report["competition_layout_mismatches"] = mismatches

    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.strict and mismatches:
        raise SystemExit("dataset audit failed: " + "; ".join(mismatches))


if __name__ == "__main__":
    main()
