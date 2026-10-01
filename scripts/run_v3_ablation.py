import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

from src.utils import load_yaml


V3_VARIANTS = [
    (
        "v2_reference",
        {
            "model.advanced.instance_embedding_head": False,
            "loss.h0_persistence": 0.0,
            "loss.instance_embedding": 0.0,
            "inference.bridge_geodesic": False,
            "inference.bridge_embedding_weight": 0.0,
            "inference.bridge_min_embedding_similarity": -1.0,
        },
    ),
    (
        "h0_only",
        {
            "model.advanced.instance_embedding_head": False,
            "loss.h0_persistence": 0.015,
            "loss.instance_embedding": 0.0,
            "inference.bridge_geodesic": False,
            "inference.bridge_embedding_weight": 0.0,
            "inference.bridge_min_embedding_similarity": -1.0,
        },
    ),
    (
        "embedding_only",
        {
            "model.advanced.instance_embedding_head": True,
            "loss.h0_persistence": 0.0,
            "loss.instance_embedding": 0.035,
            "inference.bridge_geodesic": False,
            "inference.bridge_embedding_weight": 0.35,
            "inference.bridge_min_embedding_similarity": 0.15,
        },
    ),
    (
        "geodesic_only",
        {
            "model.advanced.instance_embedding_head": False,
            "loss.h0_persistence": 0.0,
            "loss.instance_embedding": 0.0,
            "inference.bridge_geodesic": True,
            "inference.bridge_embedding_weight": 0.0,
            "inference.bridge_min_embedding_similarity": -1.0,
        },
    ),
    (
        "v3_all",
        {
            "model.advanced.instance_embedding_head": True,
            "loss.h0_persistence": 0.015,
            "loss.instance_embedding": 0.035,
            "inference.bridge_geodesic": True,
            "inference.bridge_embedding_weight": 0.35,
            "inference.bridge_min_embedding_similarity": 0.15,
        },
    ),
]


def set_path(cfg, dotted, value):
    cur = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def make_variant(base, overrides):
    cfg = copy.deepcopy(base)
    for key, value in overrides.items():
        set_path(cfg, key, value)
    return cfg


def run(cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/filanet_v3.yaml")
    ap.add_argument("--output-dir", default="artifacts/v3_ablations")
    ap.add_argument("--encoder-weights", default=None, help="same GONG SSL encoder for every trainable variant")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    args = ap.parse_args()

    base = load_yaml(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = args.folds or list(range(int(base["n_folds"])))
    manifest = []

    for name, overrides in V3_VARIANTS:
        cfg = make_variant(base, overrides)
        cfg_path = out_dir / f"{name}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        checkpoint_dir = out_dir / name / "checkpoints"
        cache_dir = out_dir / name / "oof_cache"
        metrics_path = out_dir / name / "metrics.json"
        records_path = out_dir / name / "records.csv"

        commands = []
        for fold in folds:
            cmd = [
                sys.executable,
                "scripts/train.py",
                "--config",
                str(cfg_path),
                "--fold",
                str(fold),
                "--output",
                str(checkpoint_dir),
            ]
            if args.encoder_weights:
                cmd.extend(["--encoder-weights", args.encoder_weights])
            commands.append(cmd)
        commands.extend(
            [
                [
                    sys.executable,
                    "scripts/cache_oof.py",
                    "--config",
                    str(cfg_path),
                    "--weights-pattern",
                    str(checkpoint_dir / "fold{fold}_best.pt"),
                    "--cache-dir",
                    str(cache_dir),
                ],
                [
                    sys.executable,
                    "scripts/evaluate_oof.py",
                    "--config",
                    str(cfg_path),
                    "--cache-dir",
                    str(cache_dir),
                    "--output",
                    str(metrics_path),
                    "--records-csv",
                    str(records_path),
                ],
            ]
        )
        manifest.append(
            {
                "name": name,
                "overrides": overrides,
                "config": str(cfg_path),
                "metrics": str(metrics_path),
                "records": str(records_path),
                "encoder_weights": args.encoder_weights,
                "commands": commands,
            }
        )
        if args.execute:
            for command in commands:
                run(command)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")
    if not args.execute:
        print("dry run only; execute on an authorized GPU machine with MAGFiLO data")


if __name__ == "__main__":
    main()
