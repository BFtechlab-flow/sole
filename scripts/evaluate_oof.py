"""Official-metric evaluation of the cached OOF predictions.

Primary number: micro Panoptic Quality exactly as the competition computes it
(``src/pq_official.py``). Also written: SQ/RQ, TP/FP/FN, per-fold PQ, a
bootstrap CI resampling whole observations, the IoU/Dice distributions and
the one-to-many / many-to-one relation histograms of the judging rubric, and
a per-entry CSV used by ``scripts/compare_oof_experiments.py``.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root, so `src` imports work

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from src.engine import load_records_and_folds, postprocess_params
from src.oof_eval import build_jobs, default_workers, run_jobs
from src.pq_official import aggregate_scores, bootstrap_micro_pq
from src.utils import load_yaml

CSV_FIELDS = ["entry", "stem", "file_name", "fold", "scored", "gt_count", "pred_count", "tp", "fp", "fn",
              "iou_sum", "pq", "sq", "rq", "one_to_many", "many_to_one"]


def export_official_format(records, scores, out_dir: Path):
    """Write the OOF predictions and the matching ground truth as the
    organizers' notebook expects them (``filament_id,segmentation_rle``:
    predictions ``<stem>_<k>``, ground truth ``<annotator>-<stem>_<k>``), plus
    the three rubric panels as a PNG. Run the organizers' notebook (or
    ``tools/verify_against_official_notebook.py``) on these two files to
    reproduce the OOF score with their own code."""
    import pandas as pd

    from src.evaluation import record_gt_rles
    from src.submission import write_submission

    out_dir.mkdir(parents=True, exist_ok=True)
    by_entry = {str(r["image_id"]): r for r in records}
    gt_rows, pred_rows, seen = [], [], set()
    for s in sorted(scores, key=lambda r: r["entry"]):
        for k, rle in enumerate(record_gt_rles(by_entry[s["entry"]]), 1):
            gt_rows.append({"filament_id": f"{s['entry']}_{k}", "segmentation_rle": rle["counts"].decode("ascii")})
        if s["stem"] not in seen:
            seen.add(s["stem"])
            pred_rows += [{"filament_id": f"{s['stem']}_{k}", "segmentation_rle": c}
                          for k, c in enumerate(s.get("pred_counts", []), 1)]
    cols = ["filament_id", "segmentation_rle"]
    write_submission(pd.DataFrame(gt_rows, columns=cols), out_dir / "oof_gt.csv")
    write_submission(pd.DataFrame(pred_rows, columns=cols), out_dir / "oof_pred.csv")
    try:
        from src.plots import summary_figure

        summary_figure(scores, path=out_dir / "oof_rubric_panels.png")
    except ImportError as exc:  # matplotlib missing: CSVs are still written
        print(f"rubric figure skipped: {exc}")
    print(f"official-format OOF files in {out_dir}: {len(gt_rows)} GT rows, {len(pred_rows)} prediction rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/filanet.yaml")
    ap.add_argument("--cache-dir", default="artifacts/oof_cache")
    ap.add_argument("--output", default="artifacts/oof_metrics.json")
    ap.add_argument("--records-csv", default="artifacts/oof_records.csv")
    ap.add_argument("--bootstrap", type=int, default=None)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--folds", nargs="*", type=int, default=None, help="evaluate a subset of folds")
    ap.add_argument("--export-dir", default=None,
                    help="also write oof_gt.csv / oof_pred.csv in the official format + rubric figure")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    records, folds, _, mode = load_records_and_folds(cfg)
    params = postprocess_params(cfg)
    jobs = build_jobs(records, folds, args.cache_dir, [params], cfg.get("metric"), detailed=True,
                      folds_subset=args.folds or None)
    t0 = time.time()
    scores = run_jobs(jobs, workers=args.workers, progress=lambda d, n: print(f"evaluated {d}/{n} OOF images", flush=True))[0]
    seconds = time.time() - t0

    agg = aggregate_scores(scores)
    iterations = int(args.bootstrap if args.bootstrap is not None else cfg["metric"].get("bootstrap_iterations", 2000))
    agg["bootstrap_micro_pq"] = bootstrap_micro_pq(scores, groups=[s["stem"] for s in scores],
                                                   iterations=iterations, seed=int(cfg["seed"]))
    agg["fold_pq"] = {}
    for f in (args.folds or range(int(cfg["n_folds"]))):
        part = [s for s in scores if s["fold"] == f]
        agg["fold_pq"][str(f)] = aggregate_scores(part)["pq"] if part else None
    agg.update({
        "metric": "official micro PQ (IoU > 0.5, every annotator entry scored)",
        "group_mode": mode,
        "postprocess": params,
        "oof_images": len(jobs),
        "postprocess_seconds_per_image": seconds / max(len(jobs), 1),
    })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    csv_path = Path(args.records_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for s in sorted(scores, key=lambda r: r["entry"]):
            writer.writerow(s)
    if args.export_dir:
        export_official_format(records, scores, Path(args.export_dir))
    keys = ("pq", "sq", "rq", "tp", "fp", "fn", "scored_records", "fragmentation_rate", "merge_rate")
    print(json.dumps({k: agg[k] for k in keys} | {"ci95": agg["bootstrap_micro_pq"]["ci95"], "fold_pq": agg["fold_pq"]}, indent=2))


if __name__ == "__main__":
    main()
