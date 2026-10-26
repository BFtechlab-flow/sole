"""Multi-session training helpers (Kaggle: 12 h sessions, 1 or 2 GPUs).

* ``collect_previous_checkpoints`` finds the fold checkpoints written by a
  previous session (its output attached as an input) and restores, for every
  fold, the most advanced copy.
* ``run_fold_jobs`` trains folds on several GPUs at once: one training process
  per GPU (``CUDA_VISIBLE_DEVICES``), each GPU taking the next fold when it
  is free. Each process is an ordinary ``scripts/train.py`` run, so the
  method and the results do not depend on how many GPUs were used; only the
  wall-clock time does. The same scheduler runs the GPU steps after training
  (OOF cache, test maps).
* ``restore_post_training_artifacts`` brings back the OOF cache, the cached
  test maps, the calibration memo and the rescoring feature cache of a
  previous session, but only when they were produced by exactly the current
  fold checkpoints (SHA-256).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

def logged_epochs(log_path) -> int:
    """Last epoch recorded in ``fold{k}_log.jsonl`` (0 when absent or empty)."""
    p = Path(log_path)
    if not p.is_file():
        return 0
    epoch = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                epoch = max(epoch, int(json.loads(line).get("epoch", 0)))
            except (ValueError, TypeError):
                continue
    return epoch


def find_checkpoint_dirs(roots, exclude=("competitions",)) -> list[Path]:
    """Directories below ``roots`` that contain ``fold*_last.pt`` files.

    Smoke-mode runs keep their (synthetic-data) checkpoints in
    ``artifacts/checkpoints``; the official runs use ``<repo>/checkpoints``.
    The smoke ones and those of screening experiments (any ``screening``
    folder) are never returned, so attaching such an output cannot replace
    the real checkpoints.
    """
    dirs = set()
    for root in map(Path, roots):
        if not root.is_dir():
            continue
        dirs.update(p.parent for p in root.glob("fold*_last.pt"))
        for child in root.iterdir():
            if child.is_dir() and child.name not in exclude:  # the competition data hold no checkpoint
                dirs.update(p.parent for p in child.rglob("fold*_last.pt"))
    # smoke runs keep checkpoints in artifacts/checkpoints; screening experiments under .../screening/...
    return sorted(d for d in dirs if d.parent.name != "artifacts" and "screening" not in d.parts)


def collect_previous_checkpoints(sources, dest, n_folds: int, log=print) -> dict:
    """Copy into ``dest`` the most advanced checkpoint set of every fold.

    For each fold the source whose ``fold{k}_log.jsonl`` reaches the highest
    epoch wins, and its ``best``/``last``/``log`` files are copied together
    (never mixed between sessions). A fold already further along in ``dest``
    is left untouched. Returns ``{fold: epoch}`` for the restored folds.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    restored = {}
    for fold in range(n_folds):
        here = logged_epochs(dest / f"fold{fold}_log.jsonl") if (dest / f"fold{fold}_last.pt").exists() else 0
        best_src, best_epoch = None, here
        for src in map(Path, sources):
            if src.resolve() == dest.resolve() or not (src / f"fold{fold}_last.pt").is_file():
                continue
            e = logged_epochs(src / f"fold{fold}_log.jsonl")
            if e > best_epoch:
                best_src, best_epoch = src, e
        if best_src is None:
            continue
        for name in (f"fold{fold}_best.pt", f"fold{fold}_last.pt", f"fold{fold}_log.jsonl"):
            if (best_src / name).is_file():
                shutil.copy2(best_src / name, dest / name)
            elif (dest / name).exists():
                (dest / name).unlink()  # do not keep a file of another session next to this set
        restored[fold] = best_epoch
        log(f"restored fold {fold} (epoch {best_epoch}) from {best_src}")
    return restored


def _stream(cmd, cwd, env, prefix):
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        print(prefix + line, end="", flush=True)
    return proc.wait()


def run_fold_jobs(folds, make_args, gpus, env, cwd, hours_left, min_hours: float = 0.5) -> list[str]:
    """Train ``folds`` with one process per GPU in ``gpus``.

    ``make_args(fold, remaining_hours, n_parallel)`` returns the arguments of
    ``python`` (script first). A GPU takes the next fold only while
    ``hours_left() >= min_hours``. ``gpus`` may be ``[None]`` (one process,
    device left to the default, e.g. CPU). Returns the list of failures; a
    failing fold does not stop the other GPUs.
    """
    queue = list(folds)
    lock = threading.Lock()
    failures: list[str] = []
    n_parallel = max(1, len(gpus))

    def slot(gpu):
        prefix = "" if n_parallel == 1 else f"[gpu{gpu}] "
        while True:
            with lock:
                if not queue:
                    return
                remaining = hours_left()
                if remaining < min_hours:
                    print(f"{prefix}session budget used: rerun the notebook to continue ({queue[0]} not started)",
                          flush=True)
                    return
                fold = queue.pop(0)
            args = [str(a) for a in make_args(fold, remaining, n_parallel)]
            run_env = dict(env)
            if gpu is not None:
                run_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(f"{prefix}$ python {' '.join(args)}", flush=True)
            code = _stream([sys.executable, *args], cwd, run_env, prefix)
            if code != 0:
                with lock:
                    failures.append(f"fold {fold}: exit code {code}")
                return

    threads = [threading.Thread(target=slot, args=(g,), daemon=True) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return failures


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_missing(src: Path, dst: Path) -> int:
    n = 0
    for f in src.rglob("*"):
        if f.is_file():
            target = dst / f.relative_to(src)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
                n += 1
    return n


def _oof_weights(cache_dir: Path) -> dict:
    """fold -> SHA-256 of the checkpoint that produced the cached maps (from the cache manifests)."""
    out = {}
    for m in sorted(cache_dir.glob("manifest*.json")):
        try:
            for fold, info in json.loads(m.read_text()).get("folds", {}).items():
                if info.get("weights_sha256"):
                    out[str(fold)] = info["weights_sha256"]
        except (OSError, ValueError):
            continue
    return out


def _memo_size(path: Path) -> int:
    return _memo_info(path)[0]


def _memo_info(path: Path):
    """(number of evaluated candidates, OOF weights the memo was computed on or None for older memos)."""
    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as data:
            weights = json.loads(str(data["oof_weights"])) if "oof_weights" in data.files else None
            return int(len(data["tokens"])), weights
    except Exception:  # noqa: BLE001 - unreadable memo: ignored
        return -1, None


def _fold_sha(weights) -> str:
    """SHA-256 of a fold's checkpoint, or ``sha1+sha2+...`` for a per-fold ensemble (cache_oof.py's format)."""
    if isinstance(weights, (list, tuple)):
        return "+".join(sha256_file(w) for w in weights)
    return sha256_file(weights)


def restore_post_training_artifacts(previous_artifacts, art, fold_weights, log=print, test_weights=None) -> dict:
    """Restore ``oof_cache/``, ``test_maps/`` and ``postprocess_memo.npz`` from previous sessions.

    Only data produced by the current checkpoints are restored: OOF folds whose
    manifest SHA-256 matches ``fold_weights[k]`` (a path, or a list of paths for
    a per-fold ensemble), test maps whose manifest lists exactly the checkpoints
    of the test ensemble (``test_weights``, in order; default: one per fold).
    The calibration memo carries its own signature (checked when it is loaded);
    the largest one is restored.
    """
    art = Path(art)
    current = [_fold_sha(w) for w in fold_weights]
    test_current = [sha256_file(w) for w in test_weights] if test_weights is not None else current
    summary = {"oof_folds": {}, "test_maps": 0, "memo": None}
    current_map = {str(f): sha for f, sha in enumerate(current)}

    def memo_rank(path):
        # a memo computed on the current OOF maps first, then older memos without that record, by size;
        # memos of other checkpoints are never restored (the calibration would ignore them anyway)
        size, weights = _memo_info(path)
        if weights is not None and weights != current_map:
            return None
        return (weights is not None, size)

    here = art / "postprocess_memo.npz"
    best_memo, best_rank = None, (memo_rank(here) if here.exists() else None)
    for prev in map(Path, previous_artifacts):
        if not prev.is_dir() or prev.resolve() == art.resolve():
            continue
        oof = prev / "oof_cache"
        if oof.is_dir():
            shas = _oof_weights(oof)
            good = {f: sha for f, sha in shas.items() if int(f) < len(current) and sha == current[int(f)]}
            for f in sorted(good):
                if (oof / f"fold{f}").is_dir():
                    n = _copy_missing(oof / f"fold{f}", art / "oof_cache" / f"fold{f}")
                    summary["oof_folds"][f] = summary["oof_folds"].get(f, 0) + n
            if good:
                (art / "oof_cache").mkdir(parents=True, exist_ok=True)
                manifest = {"restored_from": str(oof), "folds": {f: {"weights_sha256": good[f]} for f in good}}
                k = len(list((art / "oof_cache").glob("manifest_restored*.json")))
                (art / "oof_cache" / f"manifest_restored{k}.json").write_text(json.dumps(manifest, indent=2))
        maps = prev / "test_maps"
        if (maps / "manifest.json").is_file():
            try:
                info = json.loads((maps / "manifest.json").read_text())
            except ValueError:
                info = {}
            if info.get("weights_sha256") == test_current:
                summary["test_maps"] += _copy_missing(maps, art / "test_maps")
        feats = prev / "rescoring" / "cache"  # one sub-directory per OOF-weights signature: never stale
        if feats.is_dir():
            summary["rescoring_features"] = summary.get("rescoring_features", 0) + _copy_missing(
                feats, art / "rescoring" / "cache")
        memo = prev / "postprocess_memo.npz"
        if memo.is_file():
            rank = memo_rank(memo)
            if rank is not None and (best_rank is None or rank > best_rank):
                best_memo, best_rank = memo, rank
    best_size = best_rank[1] if best_rank else -1
    if best_memo is not None:
        art.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_memo, art / "postprocess_memo.npz")
        summary["memo"] = {"from": str(best_memo), "candidates": best_size}
    for f, n in sorted(summary["oof_folds"].items()):
        log(f"restored {n} OOF map(s) of fold {f}")
    if summary["test_maps"]:
        log(f"restored {summary['test_maps']} cached test file(s)")
    if summary.get("rescoring_features"):
        log(f"restored {summary['rescoring_features']} cached rescoring feature file(s)")
    if summary["memo"]:
        log(f"restored the calibration memo ({best_size} evaluated candidates) from {best_memo}")
    return summary


#: configuration sections that define how a model was trained (paths, data location and workers do not)
_RECIPE_KEYS = ("model", "loss", "targets", "seed", "n_folds")


def same_training_recipe(a: dict, b: dict) -> bool:
    """True when two configurations train the same model the same way (data paths and workers aside)."""
    if any(a.get(k) != b.get(k) for k in _RECIPE_KEYS):
        return False
    ta = {k: v for k, v in (a.get("train") or {}).items() if k != "workers"}
    tb = {k: v for k, v in (b.get("train") or {}).items() if k != "workers"}
    return ta == tb and (a.get("cv") or {}).get("group_mode") == (b.get("cv") or {}).get("group_mode")


def adopt_screening_checkpoints(src, dest, member_cfg: dict, n_folds: int, log=print) -> dict:
    """Reuse fold checkpoints trained by the screening notebook for an ensemble member.

    They are copied only when every checkpoint in ``src`` was trained with the
    member's exact recipe (``same_training_recipe``); the notebook then checks,
    as for any checkpoint, that it was validated on the frozen fold.
    """
    import torch

    src = Path(src)
    bests = sorted(src.glob("fold*_best.pt"))
    if not bests:
        return {}
    for b in bests:
        saved = torch.load(b, map_location="cpu", weights_only=True).get("config") or {}
        if not same_training_recipe(saved, member_cfg):
            log(f"not adopted (trained with another recipe): {b}")
            return {}
    restored = collect_previous_checkpoints([src], dest, n_folds, log=log)
    return restored
