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


def fold_assignments(records, n_folds: int = 5, group_mode: str = "observation", fold_file=None):
    """Return one fold id per annotation record without physical-image leakage.

    With ``fold_file`` the frozen assignment of that file is used (see
    ``write_fold_file``). Without it, scikit-learn's ``GroupKFold`` computes
    it: its tie-breaking is not guaranteed across library versions (the Kaggle
    CPU and GPU images gave different folds), so a multi-session run must
    freeze the assignment its checkpoints were trained with.
    """
    if fold_file:
        folds = read_fold_file(records, fold_file, n_folds)
        assert_group_isolation(records, folds, group_mode)
        return folds
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


def _stem(record) -> str:
    return Path(record["file_name"]).stem


def write_fold_file(records, folds, path, source: str = ""):
    """``stem,fold`` per observation (CSV); ``source`` is kept in a ``# comment`` line."""
    table = {}
    for record, fold in zip(records, folds):
        if table.setdefault(_stem(record), int(fold)) != int(fold):
            raise ValueError(f"observation {_stem(record)} is in two folds")
    lines = ([f"# {source}"] if source else []) + ["stem,fold"] + [f"{k},{v}" for k, v in sorted(table.items())]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_fold_file(records, path, n_folds=None):
    table = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line == "stem,fold":
            continue
        stem, fold = line.rsplit(",", 1)
        table[stem] = int(fold)
    folds = np.array([table.get(_stem(r), -1) for r in records], dtype=np.int64)
    if np.any(folds < 0):
        missing = sorted({_stem(r) for r, f in zip(records, folds) if f < 0})
        raise ValueError(f"{path}: {len(missing)} observation(s) without a fold, e.g. {missing[:3]}")
    if n_folds is not None and set(folds.tolist()) != set(range(int(n_folds))):
        raise ValueError(f"{path}: folds {sorted(set(folds.tolist()))} do not match n_folds={n_folds}")
    return folds


def folds_from_oof_cache(records, cache_dir):
    """The assignment an OOF cache was written with (``fold{k}/<stem>.npz``), or ``None``.

    ``None`` when an observation is missing from the cache or appears in two folds.
    """
    where = {}
    for d in Path(cache_dir).glob("fold*"):
        if not d.is_dir() or not d.name[4:].isdigit():
            continue
        k = int(d.name[4:])
        for f in d.glob("*.npz"):
            if where.setdefault(f.stem, k) != k:
                return None
    folds = np.array([where.get(_stem(r), -1) for r in records], dtype=np.int64)
    return None if len(folds) == 0 or np.any(folds < 0) else folds
