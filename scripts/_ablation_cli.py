"""Shared command-line driver for the ablation campaigns (see src/campaign.py)."""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root, so `src` imports work

import argparse
import copy
import json
import shlex
import subprocess
import sys
from pathlib import Path

from src.campaign import plan_variants, set_path
from src.utils import load_yaml


def cumulative_variants(base, steps):
    """Each step applies its overrides on top of the previous step."""
    current = copy.deepcopy(base)
    out = []
    for name, overrides in steps:
        for key, value in overrides.items():
            set_path(current, key, value)
        out.append((name, copy.deepcopy(current), dict(overrides)))
    return out


def independent_variants(base, steps):
    """Each step applies its overrides to the base config only."""
    out = []
    for name, overrides in steps:
        cfg = copy.deepcopy(base)
        for key, value in overrides.items():
            set_path(cfg, key, value)
        out.append((name, cfg, dict(overrides)))
    return out


def run(cmd):
    print("+", " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main(steps, default_config, default_output, cumulative=True, description=""):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=default_config)
    ap.add_argument("--output-dir", default=default_output)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--folds", nargs="*", type=int, default=None, help="screen on a subset of folds")
    ap.add_argument("--epochs", type=int, default=None, help="override train.epochs (screening)")
    ap.add_argument("--encoder-weights", default=None, help="same GONG SSL encoder for every trained variant")
    args = ap.parse_args()

    base = load_yaml(args.config)
    folds = args.folds if args.folds else list(range(int(base["n_folds"])))
    built = (cumulative_variants if cumulative else independent_variants)(base, steps)
    plans = plan_variants([(n, c) for n, c, _ in built], args.output_dir, folds, epochs=args.epochs,
                          encoder_weights=args.encoder_weights, python=sys.executable)
    for plan, (_, _, delta) in zip(plans, built):
        plan["change"] = delta
    manifest = {"config": args.config, "folds": folds, "epochs_override": args.epochs,
                "encoder_weights": args.encoder_weights, "cumulative": cumulative,
                "metric": "official micro PQ; paired bootstrap over observations", "variants": plans}
    out = Path(args.output_dir)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    trained = sum(1 for p in plans if p["reuses_checkpoints_of"] is None)
    print(f"wrote {out / 'manifest.json'}: {len(plans)} variants, {trained} need training "
          f"({trained * len(folds)} fold trainings)")
    if not args.execute:
        print("dry run only; pass --execute on a GPU machine with the official data")
        return
    for plan in plans:
        for cmd in plan["commands"]:
            run(cmd)
        if "compare_command" in plan:
            run(plan["compare_command"])
