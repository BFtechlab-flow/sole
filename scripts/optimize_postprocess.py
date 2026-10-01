import argparse
import copy
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from src.cv import assert_group_isolation, fold_assignments
from src.data import build_targets, load_coco_records
from src.oof import cache_path, load_prediction
from src.pq_official import pq_score
from src.reconstruct import reconstruct_instances
from src.search import nested_candidate_selection
from src.utils import load_yaml, resolve_path


def candidate_grid(search_space, max_candidates, seed, base):
    keys = list(search_space)
    values = [search_space[k] for k in keys]
    combos = [dict(zip(keys, items)) for items in itertools.product(*values)]
    if len(combos) > max_candidates:
        rng = random.Random(seed); combos = rng.sample(combos, max_candidates)
    canonical_base = {k: base.get(k) for k in keys}
    if canonical_base not in combos: combos.insert(0, canonical_base)
    unique = []; seen = set()
    for item in combos:
        token = tuple((k, item[k]) for k in keys)
        if token not in seen:
            seen.add(token); unique.append(item)
    return unique


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/filanet.yaml")
    ap.add_argument("--search-config", default="configs/postprocess_search.yaml")
    ap.add_argument("--cache-dir", default="artifacts/oof_cache")
    ap.add_argument("--output", default="artifacts/postprocess_search.json")
    ap.add_argument("--write-config", default="configs/filanet_tuned.yaml")
    ap.add_argument("--max-candidates", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config); search_cfg = load_yaml(args.search_config)
    max_candidates = int(args.max_candidates if args.max_candidates is not None else search_cfg.get("max_candidates", 64))
    candidates = candidate_grid(search_cfg["space"], max_candidates, cfg["seed"], cfg["inference"])
    print(f"evaluating {len(candidates)} post-processing candidates")
    root = Path(cfg["data"]["root"])
    records = load_coco_records(resolve_path(root, cfg["data"]["train_json"]))
    mode = cfg.get("cv", {}).get("group_mode", "observation")
    folds = fold_assignments(records, cfg["n_folds"], mode); assert_group_isolation(records, folds, mode)
    threshold = float(cfg["metric"].get("iou_match_threshold", 0.5))
    by_fold_file = defaultdict(list)
    for idx, (record, fold) in enumerate(zip(records, folds)):
        by_fold_file[(int(fold), record["file_name"])].append(idx)
    sums = np.zeros((len(candidates), cfg["n_folds"]), dtype=np.float64)
    counts = np.zeros(cfg["n_folds"], dtype=np.int64)
    width_scale = float(cfg.get("targets", {}).get("width_scale_px", 64.0))

    for item_no, ((fold, file_name), record_indices) in enumerate(by_fold_file.items(), 1):
        path = cache_path(args.cache_dir, fold, file_name)
        if not path.exists(): raise FileNotFoundError(f"missing OOF cache: {path}")
        pred = load_prediction(path)
        gt_sets = [build_targets(records[i], width_scale_px=width_scale)["instances"] for i in record_indices]
        counts[fold] += len(record_indices)
        for ci, candidate in enumerate(candidates):
            inf = dict(cfg["inference"]); inf.update(candidate)
            instances = reconstruct_instances(
                pred["region"], pred["centerline"], pred["boundary"], pred["distance"],
                orientation=pred.get("orientation"), width=pred.get("width"),
                instance_embedding=pred.get("instance_embedding"), **inf,
            )
            for gt in gt_sets: sums[ci, fold] += pq_score(gt, instances, threshold=threshold)["pq"]
        if item_no % 25 == 0 or item_no == len(by_fold_file):
            print(f"processed {item_no}/{len(by_fold_file)} unique OOF images")

    score_matrix = sums / np.maximum(counts[None, :], 1)
    selection = nested_candidate_selection(score_matrix); final_idx = selection["final_candidate"]
    ranking = np.argsort(-score_matrix.mean(axis=1)); trials = []
    for idx in ranking[: min(25, len(ranking))]:
        trials.append({"candidate": int(idx), "params": candidates[int(idx)], "mean_pq": float(score_matrix[idx].mean()), "fold_pq": [float(x) for x in score_matrix[idx]]})
    payload = {"n_candidates": len(candidates), "group_mode": mode, "record_counts_by_fold": counts.tolist(), "nested_selection": selection, "final_params": candidates[final_idx], "top_trials": trials}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tuned = copy.deepcopy(cfg); tuned["inference"].update(candidates[final_idx]); tuned.setdefault("tuning", {})
    tuned["tuning"].update({"source": args.output, "nested_macro_pq": selection["nested_macro_pq"], "selected_oof_macro_pq": selection["final_oof_score"]})
    Path(args.write_config).write_text(yaml.safe_dump(tuned, sort_keys=False), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
