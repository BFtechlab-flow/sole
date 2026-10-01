from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold


_OBS_RE = re.compile(r"(?P<stamp>\d{14})(?P<site>[A-Za-z]{2})")


def observation_id(file_name: str) -> str:
    """Extract the physical GONG observation id (YYYYMMDDHHMMSSII)."""
    stem = Path(file_name).stem
    match = _OBS_RE.search(stem)
    return match.group(0) if match else stem


def temporal_group(file_name: str, mode: str = "observation") -> str:
    obs = observation_id(file_name)
    if mode == "observation":
        return obs
    if mode == "day":
        return obs[:8]
    if mode == "month":
        return obs[:6]
    raise ValueError(f"unknown group mode: {mode}")


def fold_assignments(records, n_folds: int = 5, group_mode: str = "observation"):
    """Return one fold id per annotation record without physical-image leakage."""
    idx = np.arange(len(records))
    groups = np.asarray([temporal_group(r["file_name"], group_mode) for r in records])
    folds = np.full(len(records), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=int(n_folds))
    for fold, (_, va) in enumerate(splitter.split(idx, groups=groups)):
        folds[va] = fold
    if np.any(folds < 0):
        raise RuntimeError("some records were not assigned to a fold")
    return folds


def assert_group_isolation(records, folds, group_mode: str = "observation"):
    seen = {}
    for record, fold in zip(records, folds):
        group = temporal_group(record["file_name"], group_mode)
        old = seen.setdefault(group, int(fold))
        if old != int(fold):
            raise AssertionError(f"group {group} appears in folds {old} and {fold}")
    return True
