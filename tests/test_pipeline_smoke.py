"""End-to-end CPU smoke test of every command of the pipeline on SYNTHETIC data.

audit -> train 2 folds -> cache OOF -> evaluate OOF -> calibrate
post-processing -> test inference -> submission gate -> ablation dry run.
Only checks that each step runs and produces well-formed outputs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def run(args, env):
    res = subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert res.returncode == 0, f"{args}\nSTDOUT:\n{res.stdout[-3000:]}\nSTDERR:\n{res.stderr[-3000:]}"
    return res.stdout


@pytest.mark.slow
def test_full_pipeline_on_synthetic_data(tmp_path):
    from tools.make_synthetic_dataset import make_dataset

    data = make_dataset(tmp_path / "data", n_train=8, n_test=3, size=256, seed=1)
    cfg = yaml.safe_load((ROOT / "configs/smoke_cpu.yaml").read_text())
    cfg["data"]["root"] = str(data)
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    env = dict(os.environ, PYTHONHASHSEED="0", OMP_NUM_THREADS="1")
    env.pop("FILANET_DATA_ROOT", None)
    ck, art = tmp_path / "ck", tmp_path / "art"

    out = run(["scripts/audit_data.py", "--config", str(cfg_path), "--inter-annotator"], env)
    audit = json.loads(out[out.index("{"):])
    assert audit["unique_observations"] == 8
    assert "inter_annotator_official_pq" in audit

    for fold in (0, 1):
        run(["scripts/train.py", "--config", str(cfg_path), "--fold", str(fold), "--output", str(ck)], env)
        assert (ck / f"fold{fold}_best.pt").exists()
        log = [json.loads(line) for line in (ck / f"fold{fold}_log.jsonl").read_text().splitlines()]
        assert log and "pq" in log[-1]
    # resume: a second call with the same epochs must not retrain
    out = run(["scripts/train.py", "--config", str(cfg_path), "--fold", "0", "--output", str(ck)], env)
    assert "resumed fold 0" in out

    run(["scripts/cache_oof.py", "--config", str(cfg_path), "--weights-pattern", str(ck / "fold{fold}_best.pt"),
         "--cache-dir", str(art / "oof")], env)
    run(["scripts/evaluate_oof.py", "--config", str(cfg_path), "--cache-dir", str(art / "oof"),
         "--output", str(art / "oof_metrics.json"), "--records-csv", str(art / "records.csv"), "--workers", "2"], env)
    metrics = json.loads((art / "oof_metrics.json").read_text())
    assert 0.0 <= metrics["pq"] <= 1.0 and metrics["scored_records"] > 0
    records = pd.read_csv(art / "records.csv")
    assert {"entry", "stem", "fold", "iou_sum", "tp", "fp", "fn"} <= set(records.columns)

    run(["scripts/compare_oof_experiments.py", "--baseline", str(art / "records.csv"), "--candidate",
         str(art / "records.csv"), "--output", str(art / "cmp.json"), "--iterations", "100"], env)
    assert json.loads((art / "cmp.json").read_text())["micro_pq"]["delta_pq"] == 0.0

    tuned = tmp_path / "tuned.yaml"
    run(["scripts/optimize_postprocess.py", "--config", str(cfg_path), "--search-config",
         "configs/postprocess_search_smoke.yaml", "--cache-dir", str(art / "oof"), "--output",
         str(art / "search.json"), "--write-config", str(tuned), "--workers", "2"], env)
    search = json.loads((art / "search.json").read_text())
    assert {"nested_micro_pq", "final_params", "base_oof_micro_pq"} <= set(search)
    assert yaml.safe_load(tuned.read_text())["tuning"]["mode"] == "coordinate"

    sub = tmp_path / "submission.csv"
    run(["scripts/infer.py", "--config", str(tuned), "--weights", str(ck / "fold0_best.pt"), str(ck / "fold1_best.pt"),
         "--output", str(sub)], env)
    report = json.loads(sub.with_suffix(".verification.json").read_text())
    assert report["ok"] and report["runtime"]["images"] == 3
    text = sub.read_text()
    assert text.startswith("filament_id,segmentation_rle\n") and '"' not in text

    out = run(["scripts/run_ablation.py", "--config", str(cfg_path), "--output-dir", str(art / "abl"),
               "--folds", "0", "--epochs", "1"], env)
    manifest = json.loads((art / "abl" / "manifest.json").read_text())
    names = [v["name"] for v in manifest["variants"]]
    assert names[:5] == ["baseline", "plus_topology", "plus_orientation", "plus_bridge", "plus_tta"]
    by = {v["name"]: v for v in manifest["variants"]}
    assert by["plus_bridge"]["reuses_checkpoints_of"] and by["plus_bridge"]["reuses_cache_of"]
    assert by["plus_tta"]["reuses_checkpoints_of"] and not by["plus_tta"]["reuses_cache_of"]
