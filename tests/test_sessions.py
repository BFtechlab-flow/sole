"""src/sessions.py: restoring checkpoints of previous sessions and parallel fold jobs."""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.sessions import collect_previous_checkpoints, find_checkpoint_dirs, logged_epochs, run_fold_jobs


def _fold_files(d: Path, fold: int, epochs: int, tag: str):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"fold{fold}_best.pt").write_text(f"best {tag}")
    (d / f"fold{fold}_last.pt").write_text(f"last {tag}")
    (d / f"fold{fold}_log.jsonl").write_text("".join(json.dumps({"fold": fold, "epoch": e}) + "\n"
                                                     for e in range(1, epochs + 1)))


def test_logged_epochs(tmp_path):
    assert logged_epochs(tmp_path / "missing.jsonl") == 0
    _fold_files(tmp_path, 0, 7, "a")
    assert logged_epochs(tmp_path / "fold0_log.jsonl") == 7


def test_find_and_collect_most_advanced_copy(tmp_path):
    inp = tmp_path / "input"
    s1 = inp / "notebooks" / "user" / "fila-net-v1" / "sole" / "checkpoints"
    s2 = inp / "fila-net-v2" / "sole" / "checkpoints"
    _fold_files(s1, 0, 30, "s1")
    _fold_files(s1, 1, 21, "s1")
    _fold_files(s2, 1, 30, "s2")  # a later session finished fold 1
    _fold_files(inp / "competitions" / "x", 2, 30, "never")  # excluded tree
    found = find_checkpoint_dirs([inp])
    assert found == sorted([s1, s2])

    dest = tmp_path / "work" / "checkpoints"
    _fold_files(dest, 2, 3, "local")
    restored = collect_previous_checkpoints(found, dest, n_folds=5, log=lambda *_: None)
    assert restored == {0: 30, 1: 30}
    assert (dest / "fold0_last.pt").read_text() == "last s1"
    assert (dest / "fold1_best.pt").read_text() == "best s2" and (dest / "fold1_last.pt").read_text() == "last s2"
    assert (dest / "fold2_last.pt").read_text() == "last local"  # nothing better elsewhere

    # running it again changes nothing
    assert collect_previous_checkpoints(found, dest, n_folds=5, log=lambda *_: None) == {}


def test_run_fold_jobs_parallel_and_budget(tmp_path):
    script = tmp_path / "job.py"
    script.write_text("import os, sys, time\n"
                      "time.sleep(0.3)\n"
                      "open(sys.argv[1], 'w').write(os.environ.get('CUDA_VISIBLE_DEVICES', 'none'))\n")

    def make_args(fold, remaining, n_parallel):
        return [script, tmp_path / f"done{fold}.txt"]

    t0 = time.time()
    failures = run_fold_jobs([0, 1, 2, 3], make_args, gpus=[0, 1], env={}, cwd=tmp_path, hours_left=lambda: 10.0)
    elapsed = time.time() - t0
    assert failures == []
    used = sorted((tmp_path / f"done{f}.txt").read_text() for f in range(4))
    assert used == ["0", "0", "1", "1"]
    assert elapsed < 4 * 0.3 + 1.5  # ran two at a time (generous bound for slow machines)

    # no budget left: nothing starts
    for f in range(4):
        (tmp_path / f"done{f}.txt").unlink()
    assert run_fold_jobs([0, 1], make_args, gpus=[None], env={}, cwd=tmp_path, hours_left=lambda: 0.1) == []
    assert not any((tmp_path / f"done{f}.txt").exists() for f in range(2))


def test_run_fold_jobs_reports_failures(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("raise SystemExit(3)\n")
    failures = run_fold_jobs([5], lambda f, r, n: [bad], gpus=[None], env={}, cwd=tmp_path, hours_left=lambda: 5.0)
    assert failures == ["fold 5: exit code 3"]


def test_restore_post_training_artifacts_only_from_the_same_checkpoints(tmp_path):
    import numpy as np

    from src.sessions import restore_post_training_artifacts, sha256_file

    ck = tmp_path / "ck"
    ck.mkdir()
    weights = []
    for f in range(2):
        (ck / f"fold{f}_best.pt").write_bytes(f"weights {f}".encode())
        weights.append(ck / f"fold{f}_best.pt")
    sha = [sha256_file(w) for w in weights]

    prev = tmp_path / "input" / "sole" / "artifacts"
    for f in range(2):
        (prev / "oof_cache" / f"fold{f}").mkdir(parents=True)
        (prev / "oof_cache" / f"fold{f}" / "a.npz").write_text(f"oof {f}")
    (prev / "oof_cache" / "manifest.json").write_text(json.dumps(
        {"folds": {"0": {"weights_sha256": sha[0]}, "1": {"weights_sha256": "other"}}}))
    (prev / "test_maps").mkdir()
    (prev / "test_maps" / "t1.npz").write_text("map")
    (prev / "test_maps" / "manifest.json").write_text(json.dumps({"weights_sha256": sha}))
    np.savez_compressed(prev / "postprocess_memo.npz", signature=np.array("s"), tokens=np.array(["a", "b"]),
                        counts=np.zeros((2, 3, 4)), entry_folds=np.zeros(3, int))

    stale = tmp_path / "input2" / "sole" / "artifacts"
    (stale / "test_maps").mkdir(parents=True)
    (stale / "test_maps" / "t2.npz").write_text("stale map")
    (stale / "test_maps" / "manifest.json").write_text(json.dumps({"weights_sha256": ["x", "y"]}))
    np.savez_compressed(stale / "postprocess_memo.npz", signature=np.array("s"), tokens=np.array(["a"]),
                        counts=np.zeros((1, 3, 4)), entry_folds=np.zeros(3, int))

    art = tmp_path / "work" / "artifacts"
    summary = restore_post_training_artifacts([stale, prev], art, weights, log=lambda *_: None)
    assert summary["oof_folds"] == {"0": 1}
    assert (art / "oof_cache" / "fold0" / "a.npz").read_text() == "oof 0"
    assert not (art / "oof_cache" / "fold1").exists()  # produced by other weights
    restored_manifest = json.loads((art / "oof_cache" / "manifest_restored0.json").read_text())
    assert restored_manifest["folds"] == {"0": {"weights_sha256": sha[0]}}
    assert (art / "test_maps" / "t1.npz").exists() and not (art / "test_maps" / "t2.npz").exists()
    assert summary["memo"]["candidates"] == 2 and "input" in summary["memo"]["from"]
