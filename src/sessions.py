"""Multi-session training helpers (Kaggle: 12 h sessions, 1 or 2 GPUs).

* ``collect_previous_checkpoints`` finds the fold checkpoints written by a
  previous session (its output attached as an input) and restores, for every
  fold, the most advanced copy.
* ``run_fold_jobs`` trains folds on several GPUs at once: one training process
  per GPU (``CUDA_VISIBLE_DEVICES``), each GPU taking the next fold when it
  is free. Each process is an ordinary ``scripts/train.py`` run, so the
  method and the results do not depend on how many GPUs were used; only the
  wall-clock time does.
"""
from __future__ import annotations

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
    """Directories below ``roots`` that contain ``fold*_last.pt`` files."""
    dirs = set()
    for root in map(Path, roots):
        if not root.is_dir():
            continue
        dirs.update(p.parent for p in root.glob("fold*_last.pt"))
        for child in root.iterdir():
            if child.is_dir() and child.name not in exclude:  # the competition data hold no checkpoint
                dirs.update(p.parent for p in child.rglob("fold*_last.pt"))
    return sorted(dirs)


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
                    print(f"{prefix}session budget used: rerun the notebook to continue with fold {queue[0]}", flush=True)
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
