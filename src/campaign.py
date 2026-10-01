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
