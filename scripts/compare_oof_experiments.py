import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.campaign import paired_bootstrap


def align_records(baseline_path, candidate_path):
    a = pd.read_csv(baseline_path)
    b = pd.read_csv(candidate_path)
    required = {"record_index", "fold", "pq", "sq", "rq", "tp", "fp", "fn"}
    missing_a = required - set(a.columns)
    missing_b = required - set(b.columns)
    if missing_a or missing_b:
        raise ValueError(f"missing columns baseline={sorted(missing_a)} candidate={sorted(missing_b)}")
    if a["record_index"].duplicated().any() or b["record_index"].duplicated().any():
        raise ValueError("record_index must be unique in each OOF records CSV")
    merged = a.merge(b, on="record_index", suffixes=("_baseline", "_candidate"), how="inner")
    if len(merged) != len(a) or len(merged) != len(b):
        raise ValueError(
            f"OOF record sets differ: baseline={len(a)} candidate={len(b)} aligned={len(merged)}"
        )
    if not np.array_equal(merged["fold_baseline"].to_numpy(), merged["fold_candidate"].to_numpy()):
        raise ValueError("fold assignments differ between experiments")
    return merged.sort_values("record_index").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--output", default="artifacts/supervised_campaign/comparison.json")
    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    frame = align_records(args.baseline, args.candidate)
    report = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "records": int(len(frame)),
        "metrics": {},
        "folds": {},
        "count_deltas": {},
    }
    for metric in ("pq", "sq", "rq"):
        report["metrics"][metric] = paired_bootstrap(
            frame[f"{metric}_baseline"].to_numpy(),
            frame[f"{metric}_candidate"].to_numpy(),
            iterations=args.iterations,
            seed=args.seed,
        )

    for fold in sorted(frame["fold_baseline"].unique()):
        part = frame[frame["fold_baseline"] == fold]
        report["folds"][str(int(fold))] = {
            "records": int(len(part)),
            "baseline_pq": float(part["pq_baseline"].mean()),
            "candidate_pq": float(part["pq_candidate"].mean()),
            "delta_pq": float((part["pq_candidate"] - part["pq_baseline"]).mean()),
        }

    for metric in ("tp", "fp", "fn"):
        report["count_deltas"][metric] = int(
            frame[f"{metric}_candidate"].sum() - frame[f"{metric}_baseline"].sum()
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
