import argparse
import json
import shlex
import subprocess
from pathlib import Path

from src.campaign import experiment_commands
from src.utils import load_yaml


def run(command):
    print("+", " ".join(shlex.quote(str(x)) for x in command), flush=True)
    subprocess.run(command, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/filanet.yaml")
    ap.add_argument("--output-dir", default="artifacts/supervised_campaign")
    ap.add_argument("--encoder-weights", default="artifacts/pretrain/gong_ssl_imagenet.pt")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bootstrap-iterations", type=int, default=5000)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_folds = int(cfg["n_folds"])

    experiments = [
        {
            "name": "v2_imagenet",
            "initialization": "ImageNet weights from configured timm encoder",
            "encoder_weights": None,
        },
        {
            "name": "v2_imagenet_gong",
            "initialization": "same ImageNet encoder, then public NOAA/GONG H-alpha SSL adaptation",
            "encoder_weights": args.encoder_weights,
        },
    ]

    manifest = {
        "config": args.config,
        "n_folds": n_folds,
        "epochs_override": args.epochs,
        "comparison_policy": (
            "paired leakage-safe OOF; identical architecture, folds, seed and post-processing. "
            "Only encoder initialization differs between the first two experiments."
        ),
        "experiments": [],
    }

    if args.execute and not Path(args.encoder_weights).exists():
        raise FileNotFoundError(
            f"missing GONG SSL checkpoint {args.encoder_weights}; generate/download an ImageNet-initialized SSL checkpoint first"
        )

    for exp in experiments:
        commands = experiment_commands(
            args.config,
            out,
            exp["name"],
            n_folds=n_folds,
            encoder_weights=exp["encoder_weights"],
            epochs=args.epochs,
        )
        manifest["experiments"].append({**exp, "commands": commands})
        if args.execute:
            for command in commands:
                run(command)

    baseline_records = out / "v2_imagenet" / "records.csv"
    gong_records = out / "v2_imagenet_gong" / "records.csv"
    compare_command = [
        "python", "scripts/compare_oof_experiments.py",
        "--baseline", str(baseline_records),
        "--candidate", str(gong_records),
        "--output", str(out / "imagenet_vs_gong.json"),
        "--iterations", str(args.bootstrap_iterations),
        "--seed", str(cfg["seed"]),
    ]
    manifest["comparison_command"] = compare_command

    innovation_command = [
        "python", "scripts/run_innovation_ablation.py",
        "--config", args.config,
        "--output-dir", str(out / "innovation_ablations_gong"),
        "--encoder-weights", args.encoder_weights,
    ]
    if args.epochs is not None:
        innovation_command += ["--epochs", str(args.epochs)]
    innovation_command += ["--execute"]
    manifest["innovation_ablation_command"] = innovation_command

    manifest_path = out / "campaign_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")
    if args.execute:
        run(compare_command)
        print("baseline vs GONG comparison complete")
        print("innovation ablation is intentionally NOT auto-started; inspect the paired SSL result first")
    else:
        print("dry run only; use --execute on an authorized GPU machine with official MAGFiLO data")
        print("after paired comparison, run the innovation_ablation_command from the manifest")


if __name__ == "__main__":
    main()
