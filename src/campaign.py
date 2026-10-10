from __future__ import annotations

from pathlib import Path

import numpy as np


def paired_bootstrap(baseline, candidate, iterations: int = 5000, seed: int = 2026) -> dict:
    """Paired bootstrap of per-record metric deltas.

    Both arrays must describe the exact same validation records in the same order.
    This is intentionally paired so image difficulty cancels when comparing two
    initializations or model variants.
    """
    a = np.asarray(baseline, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("baseline and candidate must be 1-D arrays with identical shape")
    if len(a) == 0:
        raise ValueError("paired bootstrap requires at least one record")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    delta = b - a
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    n = len(delta)
    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(delta[idx].mean())
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {
        "records": int(n),
        "baseline_mean": float(a.mean()),
        "candidate_mean": float(b.mean()),
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "positive_record_fraction": float(np.mean(delta > 0)),
        "equal_record_fraction": float(np.mean(delta == 0)),
        "bootstrap_ci95": [float(lo), float(hi)],
        "bootstrap_probability_positive": float(np.mean(samples > 0)),
        "iterations": int(iterations),
        "seed": int(seed),
    }


def experiment_commands(
    config: str,
    output_dir: str | Path,
    name: str,
    n_folds: int,
    encoder_weights: str | None = None,
    epochs: int | None = None,
) -> list[list[str]]:
    """Build the complete leakage-safe 5-fold train -> OOF -> evaluate command chain."""
    out = Path(output_dir) / name
    checkpoints = out / "checkpoints"
    cache = out / "oof_cache"
    metrics = out / "metrics.json"
    records = out / "records.csv"
    commands: list[list[str]] = []
    for fold in range(int(n_folds)):
        cmd = [
            "python", "scripts/train.py", "--config", str(config),
            "--fold", str(fold), "--output", str(checkpoints),
        ]
        if epochs is not None:
            cmd += ["--epochs", str(int(epochs))]
        if encoder_weights:
            cmd += ["--encoder-weights", str(encoder_weights)]
        commands.append(cmd)
    commands.append([
        "python", "scripts/cache_oof.py", "--config", str(config),
        "--weights-pattern", str(checkpoints / "fold{fold}_best.pt"),
        "--cache-dir", str(cache),
    ])
    commands.append([
        "python", "scripts/evaluate_oof.py", "--config", str(config),
        "--cache-dir", str(cache), "--output", str(metrics),
        "--records-csv", str(records),
    ])
    return commands


# --------------------------------------------------------------------------- ablation planning
INFERENCE_ONLY_SECTIONS = ("inference", "metric", "tuning")
PREDICTION_KEYS = ("tile", "overlap", "tta", "amp")


def set_path(cfg: dict, dotted: str, value) -> None:
    cur = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def _training_view(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k not in INFERENCE_ONLY_SECTIONS}


def _prediction_view(cfg: dict) -> dict:
    inf = cfg.get("inference", {})
    return {k: inf.get(k) for k in PREDICTION_KEYS}


def plan_variants(variants, out_dir, folds, epochs=None, encoder_weights=None, python="python"):
    """Commands for a list of ``(name, cfg)`` variants, reusing work when possible.

    * same training-relevant config as an earlier variant -> reuse its checkpoints;
    * additionally same tiling/TTA -> reuse its OOF cache (post-processing only).

    Inference-only ablations (graph bridge, geodesic repair, TTA, thresholds)
    therefore cost no GPU training at all. ``folds`` restricts training, OOF
    caching and evaluation to a subset (screening runs).
    """
    import copy

    import yaml

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plans, history = [], []
    fold_args = [str(int(f)) for f in folds]
    for name, cfg in variants:
        cfg = copy.deepcopy(cfg)
        cfg_path = out / f"{name}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        ckpt_dir, cache_dir = out / name / "checkpoints", out / name / "oof_cache"
        reuse_ckpt = reuse_cache = None
        for prev in reversed(history):
            if _training_view(prev["cfg"]) == _training_view(cfg):
                reuse_ckpt = prev["checkpoints"]
                if _prediction_view(prev["cfg"]) == _prediction_view(cfg):
                    reuse_cache = prev["cache"]
                break
        commands = []
        if reuse_ckpt is None:
            for fold in folds:
                cmd = [python, "scripts/train.py", "--config", str(cfg_path), "--fold", str(int(fold)),
                       "--output", str(ckpt_dir)]
                if epochs is not None:
                    cmd += ["--epochs", str(int(epochs))]
                if encoder_weights:
                    cmd += ["--encoder-weights", str(encoder_weights)]
                commands.append(cmd)
        else:
            ckpt_dir = Path(reuse_ckpt)
        if reuse_cache is None:
            commands.append([python, "scripts/cache_oof.py", "--config", str(cfg_path),
                             "--weights-pattern", str(ckpt_dir / "fold{fold}_best.pt"),
                             "--cache-dir", str(cache_dir), "--folds", *fold_args])
        else:
            cache_dir = Path(reuse_cache)
        commands.append([python, "scripts/evaluate_oof.py", "--config", str(cfg_path),
                         "--cache-dir", str(cache_dir), "--folds", *fold_args,
                         "--output", str(out / name / "metrics.json"),
                         "--records-csv", str(out / name / "records.csv")])
        plan = {"name": name, "config": str(cfg_path), "checkpoints": str(ckpt_dir), "cache": str(cache_dir),
                "reuses_checkpoints_of": reuse_ckpt, "reuses_cache_of": reuse_cache,
                "metrics": str(out / name / "metrics.json"), "records": str(out / name / "records.csv"),
                "commands": commands}
        plans.append(plan)
        history.append({"cfg": cfg, "checkpoints": str(ckpt_dir), "cache": str(cache_dir)})
    for prev, cur in zip(plans, plans[1:]):
        cur["compare_command"] = [python, "scripts/compare_oof_experiments.py", "--baseline", prev["records"],
                                  "--candidate", cur["records"], "--output",
                                  str(out / f"compare_{prev['name']}__{cur['name']}.json")]
    return plans
