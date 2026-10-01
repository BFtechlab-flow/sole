import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

from src.utils import load_yaml


ABLATIONS = [
    (
        "baseline",
        {
            "loss.topology": 0.0,
            "loss.orientation": 0.0,
            "inference.bridge_max_distance": 0,
            "inference.tta": False,
            "train.validation_tta": False,
        },
    ),
    ("plus_topology", {"loss.topology": 0.05}),
    ("plus_orientation", {"loss.orientation": 0.08}),
    ("plus_bridge", {"inference.bridge_max_distance": 28}),
    ("plus_tta", {"inference.tta": True, "train.validation_tta": True}),
]


def set_path(cfg, dotted, value):
    cur = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def build_variants(base):
    current = copy.deepcopy(base)
    out = []
    for name, overrides in ABLATIONS:
        for key, value in overrides.items():
            set_path(current, key, value)
        out.append((name, copy.deepcopy(current), dict(overrides)))
    return out


def run(cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/filanet.yaml")
    ap.add_argument("--output-dir", default="artifacts/ablations")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    args = ap.parse_args()

    base = load_yaml(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = build_variants(base)
    manifest = []

    for name, cfg, delta in variants:
        cfg_path = out_dir / f"{name}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        checkpoint_dir = out_dir / name / "checkpoints"
        cache_dir = out_dir / name / "oof_cache"
        metrics_path = out_dir / name / "metrics.json"
        folds = args.folds or list(range(cfg["n_folds"]))
        commands = []
        for fold in folds:
            commands.append([
                sys.executable, "scripts/train.py", "--config", str(cfg_path),
                "--fold", str(fold), "--output", str(checkpoint_dir),
            ])
        commands.append([
            sys.executable, "scripts/cache_oof.py", "--config", str(cfg_path),
            "--weights-pattern", str(checkpoint_dir / "fold{fold}_best.pt"),
            "--cache-dir", str(cache_dir),
        ])
        commands.append([
            sys.executable, "scripts/evaluate_oof.py", "--config", str(cfg_path),
            "--cache-dir", str(cache_dir), "--output", str(metrics_path),
            "--records-csv", str(out_dir / name / "records.csv"),
        ])
        manifest.append({
            "name": name,
            "single_change": delta,
            "config": str(cfg_path),
            "metrics": str(metrics_path),
            "commands": commands,
        })
        if args.execute:
            for command in commands:
                run(command)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")
    if not args.execute:
        print("dry run only; pass --execute on a machine with the MAGFiLO data and GPU")


if __name__ == "__main__":
    main()
