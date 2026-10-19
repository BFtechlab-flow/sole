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
    The smoke ones are never returned, so attaching the output of a smoke run
    cannot replace real checkpoints.
    """
    dirs = set()
    for root in map(Path, roots):
        if not root.is_dir():
            continue
        dirs.update(p.parent for p in root.glob("fold*_last.pt"))
        for child in root.iterdir():
            if child.is_dir() and child.name not in exclude:  # the competition data hold no checkpoint
                dirs.update(p.parent for p in child.rglob("fold*_last.pt"))
    return sorted(d for d in dirs if d.parent.name != "artifacts")


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
    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as data:
            return int(len(data["tokens"]))
    except Exception:  # noqa: BLE001 - unreadable memo: ignored
        return -1


def restore_post_training_artifacts(previous_artifacts, art, fold_weights, log=print) -> dict:
    """Restore ``oof_cache/``, ``test_maps/`` and ``postprocess_memo.npz`` from previous sessions.

    Only data produced by the current checkpoints are restored: OOF folds whose
    manifest SHA-256 matches ``fold_weights[k]``, test maps whose manifest lists
    exactly the current fold checkpoints. The calibration memo carries its own
    signature (checked when it is loaded); the largest one is restored.
    """
    art = Path(art)
    current = [sha256_file(w) for w in fold_weights]
    summary = {"oof_folds": {}, "test_maps": 0, "memo": None}
    best_memo, best_size = None, _memo_size(art / "postprocess_memo.npz") if (art / "postprocess_memo.npz").exists() else -1
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
            if info.get("weights_sha256") == current:
                summary["test_maps"] += _copy_missing(maps, art / "test_maps")
        feats = prev / "rescoring" / "cache"  # one sub-directory per OOF-weights signature: never stale
        if feats.is_dir():
            summary["rescoring_features"] = summary.get("rescoring_features", 0) + _copy_missing(
                feats, art / "rescoring" / "cache")
        memo = prev / "postprocess_memo.npz"
        if memo.is_file():
            size = _memo_size(memo)
            if size > best_size:
                best_memo, best_size = memo, size
    if best_memo is not None:
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
