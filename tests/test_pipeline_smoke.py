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

    # resumable calibration: a memo replays evaluated candidates; partial memos resume to the same result
    import numpy as np

    memo = art / "memo.npz"
    search_args = ["scripts/optimize_postprocess.py", "--config", str(cfg_path), "--search-config",
                   "configs/postprocess_search_smoke.yaml", "--cache-dir", str(art / "oof"), "--write-config",
                   str(tmp_path / "tuned_memo.yaml"), "--workers", "2", "--memo", str(memo)]
    run(search_args + ["--output", str(art / "s0.json"), "--max-hours", "0.000001"], env)
    assert json.loads((art / "s0.json").read_text())["complete"] is False
    run(search_args + ["--output", str(art / "s1.json")], env)
    s1 = json.loads((art / "s1.json").read_text())
    keys = ("nested_micro_pq", "final_params", "base_oof_micro_pq", "final_oof_micro_pq", "outer_selections")
    assert s1["complete"] and all(s1[k] == search[k] for k in keys)
    with np.load(memo) as d:  # keep only half of the evaluated candidates, as after an interrupted session
        half = len(d["tokens"]) // 2
        np.savez_compressed(memo, signature=d["signature"], tokens=d["tokens"][:half], counts=d["counts"][:half],
                            entry_folds=d["entry_folds"])
    out = run(search_args + ["--output", str(art / "s2.json")], env)
    s2 = json.loads((art / "s2.json").read_text())
    assert f"restored {half} evaluated candidate(s)" in out
    assert s2["complete"] and all(s2[k] == search[k] for k in keys)
    out = run(search_args + ["--output", str(art / "s3.json"), "--max-hours", "0.000001"], env)  # pure replay
    s3 = json.loads((art / "s3.json").read_text())
    assert s3["complete"] and all(s3[k] == search[k] for k in keys) and "new candidate(s)" not in out

    sub = tmp_path / "submission.csv"
    weights = [str(ck / "fold0_best.pt"), str(ck / "fold1_best.pt")]
    run(["scripts/infer.py", "--config", str(tuned), "--weights", *weights, "--output", str(sub),
         "--maps-dir", str(art / "test_maps")], env)
    report = json.loads(sub.with_suffix(".verification.json").read_text())
    assert report["ok"] and report["runtime"]["images"] == 3
    assert report["runtime"]["images_predicted_by_the_networks"] == 3
    text = sub.read_text()
    assert text.startswith("filament_id,segmentation_rle\n") and '"' not in text
    # the same submission from the cached maps alone (no network), and from a fresh prediction
    sub_maps, sub_fresh = tmp_path / "submission_maps.csv", tmp_path / "submission_fresh.csv"
    run(["scripts/infer.py", "--config", str(tuned), "--maps-dir", str(art / "test_maps"), "--output",
         str(sub_maps)], env)
    run(["scripts/infer.py", "--config", str(tuned), "--weights", *weights, "--output", str(sub_fresh)], env)
    assert sub_maps.read_text() == text == sub_fresh.read_text()
    assert json.loads(sub_maps.with_suffix(".verification.json").read_text())["runtime"][
        "images_post_processed_from_cached_maps"] == 3

    # learned rescoring: resumable extraction, nested estimate, then the same instances from maps or a fresh run
    resc_args = ["scripts/train_rescorer.py", "--config", str(tuned), "--search-json", str(art / "search.json"),
                 "--cache-dir", str(art / "oof"), "--output-dir", str(art / "rescoring"), "--workers", "2"]
    run(resc_args + ["--max-hours", "0.000001"], env)
    assert json.loads((art / "rescoring" / "rescoring.json").read_text())["complete"] is False
    rescored = tmp_path / "tuned_rescored.yaml"
    run(resc_args + ["--write-config", str(rescored)], env)
    rep = json.loads((art / "rescoring" / "rescoring.json").read_text())
    assert rep["complete"] and rep["reconstruction_settings_nested"]
    assert {"nested_micro_pq_with_rescoring", "nested_micro_pq_without_rescoring", "final_threshold",
            "oracle_filtering_ceiling_micro_pq"} <= set(rep)
    assert rep["oracle_filtering_ceiling_micro_pq"] >= rep["all_instances_micro_pq"] - 1e-12
    rcfg = yaml.safe_load(rescored.read_text())
    assert rcfg["rescoring"]["enabled"] == (rep["nested_micro_pq_with_rescoring"] > rep["nested_micro_pq_without_rescoring"])
    rcfg["rescoring"]["enabled"] = True  # exercise the inference hook whatever the synthetic result
    rescored.write_text(yaml.safe_dump(rcfg))
    sub_r_maps, sub_r_fresh = tmp_path / "submission_r_maps.csv", tmp_path / "submission_r_fresh.csv"
    run(["scripts/infer.py", "--config", str(rescored), "--maps-dir", str(art / "test_maps"), "--output",
         str(sub_r_maps)], env)
    run(["scripts/infer.py", "--config", str(rescored), "--weights", *weights, "--output", str(sub_r_fresh)], env)
    assert sub_r_maps.read_text() == sub_r_fresh.read_text()
    assert len(sub_r_maps.read_text().splitlines()) <= len(text.splitlines())

    out = run(["scripts/run_ablation.py", "--config", str(cfg_path), "--output-dir", str(art / "abl"),
               "--folds", "0", "--epochs", "1"], env)
    manifest = json.loads((art / "abl" / "manifest.json").read_text())
    names = [v["name"] for v in manifest["variants"]]
    assert names[:5] == ["baseline", "plus_topology", "plus_orientation", "plus_bridge", "plus_tta"]
    by = {v["name"]: v for v in manifest["variants"]}
    assert by["plus_bridge"]["reuses_checkpoints_of"] and by["plus_bridge"]["reuses_cache_of"]
    assert by["plus_tta"]["reuses_checkpoints_of"] and not by["plus_tta"]["reuses_cache_of"]
