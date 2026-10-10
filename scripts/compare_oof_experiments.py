"""Paired comparison of two OOF experiments with the official micro PQ.

Both ``records.csv`` files (from ``scripts/evaluate_oof.py``) must contain the
same annotator entries with the same folds. Whole observations are resampled
together (an observation's 1-3 entries are not independent), and for every
bootstrap draw both experiments' micro PQ are recomputed from summed counts.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root, so `src` imports work

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.pq_official import paired_bootstrap_micro_pq

REQUIRED = {"entry", "stem", "fold", "scored", "tp", "fp", "fn", "iou_sum"}


def load(path):
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)} (re-run scripts/evaluate_oof.py)")
    if df["entry"].duplicated().any():
        raise ValueError(f"{path}: duplicated entries")
    return df


def align_records(baseline_path, candidate_path):
    a, b = load(baseline_path), load(candidate_path)
    merged = a.merge(b, on="entry", suffixes=("_baseline", "_candidate"), how="inner")
    if len(merged) != len(a) or len(merged) != len(b):
        raise ValueError(f"OOF entry sets differ: baseline={len(a)} candidate={len(b)} aligned={len(merged)}")
    if not np.array_equal(merged["fold_baseline"].to_numpy(), merged["fold_candidate"].to_numpy()):
        raise ValueError("fold assignments differ between experiments")
    return merged.sort_values("entry").reset_index(drop=True)


def _scores(frame, suffix):
    return [{"scored": bool(r[f"scored_{suffix}"]), "iou_sum": float(r[f"iou_sum_{suffix}"]),
             "tp": int(r[f"tp_{suffix}"]), "fp": int(r[f"fp_{suffix}"]), "fn": int(r[f"fn_{suffix}"])}
            for _, r in frame.iterrows()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--output", default="artifacts/comparison.json")
    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    frame = align_records(args.baseline, args.candidate)
    groups = frame["stem_baseline"].tolist()
    report = {"baseline": args.baseline, "candidate": args.candidate, "entries": int(len(frame))}
    report["micro_pq"] = paired_bootstrap_micro_pq(_scores(frame, "baseline"), _scores(frame, "candidate"),
                                                   groups, iterations=args.iterations, seed=args.seed)
    report["folds"] = {}
    for fold in sorted(frame["fold_baseline"].unique()):
        part = frame[frame["fold_baseline"] == fold]
        sa, sb = _scores(part, "baseline"), _scores(part, "candidate")
        res = paired_bootstrap_micro_pq(sa, sb, part["stem_baseline"].tolist(), iterations=200, seed=args.seed)
        report["folds"][str(int(fold))] = {"baseline_pq": res["baseline_pq"], "candidate_pq": res["candidate_pq"],
                                           "delta_pq": res["delta_pq"]}
    report["count_deltas"] = {m: int(frame[f"{m}_candidate"].sum() - frame[f"{m}_baseline"].sum())
                              for m in ("tp", "fp", "fn")}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
