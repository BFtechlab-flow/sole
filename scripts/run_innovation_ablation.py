import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

from src.utils import load_yaml


INNOVATIONS = [
    ("baseline_v2", {
        "model.advanced.enabled": True,
        "model.advanced.limb_gate": False,
        "model.advanced.multiscale_context": False,
        "model.advanced.frequency_branch": False,
        "model.advanced.width_head": False,
        "model.advanced.curvature_head": False,
        "model.advanced.endpoint_head": False,
        "model.advanced.junction_head": False,
        "model.advanced.uncertainty_head": False,
        "loss.width": 0.0,
        "loss.curvature": 0.0,
        "loss.endpoint": 0.0,
        "loss.junction": 0.0,
        "loss.centerline_ce": 0.0,
        "loss.radius_balance": 0.0,
        "loss.uncertainty_region": 0.0,
        "inference.bridge_width_weight": 0.0,
    }),
    ("01_limb_gate", {"model.advanced.limb_gate": True}),
    ("02_multiscale_context", {"model.advanced.multiscale_context": True}),
    ("03_frequency_branch", {"model.advanced.frequency_branch": True}),
    ("04_physical_width", {
        "model.advanced.width_head": True,
        "loss.width": 0.05,
        "inference.bridge_width_weight": 0.35,
    }),
    ("05_curvature", {
        "model.advanced.curvature_head": True,
        "loss.curvature": 0.025,
    }),
    ("06_endpoint", {
        "model.advanced.endpoint_head": True,
        "loss.endpoint": 0.035,
    }),
    ("07_junction", {
        "model.advanced.junction_head": True,
        "loss.junction": 0.020,
    }),
    ("08_centerline_ce", {"loss.centerline_ce": 0.040}),
    ("09_radius_balance", {"loss.radius_balance": 0.040}),
    ("10_uncertainty", {
        "model.advanced.uncertainty_head": True,
        "loss.uncertainty_region": 0.030,
    }),
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
    for name, overrides in INNOVATIONS:
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
    ap.add_argument("--output-dir", default="artifacts/innovation_ablations")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument(
        "--encoder-weights",
        default=None,
        help="optional shared GONG SSL checkpoint; keeps initialization constant across all innovation variants",
    )
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
            cmd = [
                sys.executable, "scripts/train.py", "--config", str(cfg_path),
                "--fold", str(fold), "--output", str(checkpoint_dir),
            ]
            if args.epochs is not None:
                cmd += ["--epochs", str(args.epochs)]
            if args.encoder_weights:
                cmd += ["--encoder-weights", str(args.encoder_weights)]
            commands.append(cmd)
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
            "encoder_weights": args.encoder_weights,
            "commands": commands,
        })
        if args.execute:
            for command in commands:
                run(command)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")
    if not args.execute:
        print("dry run only; pass --execute with official MAGFiLO data + GPU")


if __name__ == "__main__":
    main()
