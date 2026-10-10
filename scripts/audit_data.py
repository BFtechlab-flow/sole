"""Audit of the official training data + human (inter-annotator) baseline.

Checks the layout and counts announced for the competition (707 observations,
1,154 annotator records, 8,199 filaments, 2048x2048 images), the fold
isolation, and the annotation edge cases that matter for the official metric:
records without filaments (never scored), degenerate or empty polygons,
overlapping filaments inside one record.

``--inter-annotator`` scores every annotator of a multi-annotated observation
against the other annotators with the official micro PQ: the agreement
between experts, i.e. a practical ceiling for any model, to quote in the
technical report.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root, so `src` imports work

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from src.cv import assert_group_isolation, fold_assignments, observation_id
from src.data import load_coco_records
from src.evaluation import record_gt_rles
from src.pq_official import aggregate_scores, pq_score
from src.utils import data_paths, load_yaml

EXPECTED = {
    "unique_observations": 707,
    "annotation_records": 1154,
    "filament_instances": 8199,
    "image_shape": (2048, 2048),
}


def inter_annotator(records):
    by_stem = defaultdict(list)
    for r in records:
        by_stem[r["stem"]].append(r)
    scores = []
    for stem, recs in by_stem.items():
        if len(recs) < 2:
            continue
        rles = [record_gt_rles(r) for r in recs]
        for i in range(len(recs)):            # annotator i plays the "model"
            for j in range(len(recs)):        # every other annotator is a GT entry
                if i != j:
                    s = pq_score(rles[j], rles[i], shape=(recs[j]["height"], recs[j]["width"]))
                    s["stem"] = stem
                    scores.append(s)
    return scores


def day_statistics(train_stems, test_stems):
    """How much the same Sun is seen several times (same UT day) within the
    training set and between the training and test sets.

    GONG sites observe the same disk minutes apart and filaments live for days,
    so same-day observations are near-duplicates: with ``cv.group_mode:
    observation`` they can fall in different folds and inflate the OOF score.
    ``day`` grouping keeps them together (the default here).
    """
    from datetime import datetime, timedelta

    def day(stem):
        return observation_id(stem)[:8]

    train_days = Counter(day(s) for s in train_stems)
    out = {
        "train_days": len(train_days),
        "train_max_observations_per_day": max(train_days.values()) if train_days else 0,
        "train_observations_sharing_their_day": int(sum(n for n in train_days.values() if n > 1)),
    }
    if test_stems:
        def near(d, k):
            t = datetime.strptime(d, "%Y%m%d")
            return {(t + timedelta(days=x)).strftime("%Y%m%d") for x in range(-k, k + 1)}

        same = sum(day(s) in train_days for s in test_stems)
        within1 = sum(bool(near(day(s), 1) & train_days.keys()) for s in test_stems)
        out.update({
            "test_observations": len(test_stems),
            "test_observations_with_train_on_same_day": int(same),
            "test_observations_with_train_within_1_day": int(within1),
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/filanet.yaml")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--inter-annotator", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    root, json_path, image_dir, test_dir = data_paths(cfg)
    records = load_coco_records(json_path)
    stems = Counter(r["stem"] for r in records)
    n_inst = int(sum(len(r["annotations"]) for r in records))
    sizes = sorted({(r["height"], r["width"]) for r in records})
    missing = sorted({r["file_name"] for r in records if not (image_dir / r["file_name"]).exists()})
    mode = cfg.get("cv", {}).get("group_mode", "observation")
    folds = fold_assignments(records, cfg["n_folds"], mode)
    assert_group_isolation(records, folds, mode)

    degenerate, empty, overlapping_records, overlap_pairs = 0, 0, 0, 0
    areas, categories, per_record = [], Counter(), []
    for r in records:
        h, w = r["height"], r["width"]
        per_record.append(len(r["annotations"]))
        for a in r["annotations"]:
            categories[a.get("category_id")] += 1
            seg = a["segmentation"]
            if isinstance(seg, list) and any(len(p) < 6 for p in seg):
                degenerate += 1
        rles = record_gt_rles(r)
        if rles:
            ar = mask_utils.area(rles)
            areas += ar.tolist()
            empty += int((ar == 0).sum())
            if len(rles) > 1:
                iou = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)))
                np.fill_diagonal(iou, 0)
                pairs = int((np.triu(iou) > 0).sum())
                overlap_pairs += pairs
                overlapping_records += int(pairs > 0)

    test_files = sorted(p.stem for p in Path(test_dir).iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png")) if Path(test_dir).exists() else []
    annotators = Counter(stems.values())
    report = {
        "data_root": str(root),
        "annotation_records": len(records),
        "unique_observations": len(stems),
        "observations_by_number_of_annotators": {str(k): v for k, v in sorted(annotators.items())},
        "filament_instances": n_inst,
        "records_without_filaments_not_scored_officially": int(sum(n == 0 for n in per_record)),
        "filaments_per_record_mean": float(np.mean(per_record)) if per_record else 0.0,
        "category_counts": {str(k): v for k, v in sorted(categories.items(), key=lambda kv: str(kv[0]))},
        "degenerate_polygons_lt_3_points": degenerate,
        "polygons_rasterising_to_empty_masks": empty,
        "records_with_overlapping_filaments": overlapping_records,
        "overlapping_filament_pairs": overlap_pairs,
        "filament_area_px": {"min": int(min(areas)) if areas else None,
                             "median": float(np.median(areas)) if areas else None,
                             "max": int(max(areas)) if areas else None},
        "sizes": sizes,
        "missing_images": missing,
        "test_images": len(test_files),
        "fold_record_counts": {str(f): int(np.sum(folds == f)) for f in range(cfg["n_folds"])},
        "group_mode": mode,
        "same_day_observations": day_statistics(list(stems), test_files),
    }
    if args.inter_annotator:
        scores = inter_annotator(records)
        agg = aggregate_scores(scores)
        report["inter_annotator_official_pq"] = {k: agg[k] for k in (
            "pq", "sq", "rq", "tp", "fp", "fn", "scored_records", "fragmentation_rate", "merge_rate",
            "matched_iou_distribution", "overlap_dice_distribution")}

    mismatches = []
    for key in ("annotation_records", "unique_observations"):
        if report[key] != EXPECTED[key]:
            mismatches.append(f"{key}={report[key]} expected={EXPECTED[key]}")
    if n_inst != EXPECTED["filament_instances"]:
        mismatches.append(f"filament_instances={n_inst} expected={EXPECTED['filament_instances']}")
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
