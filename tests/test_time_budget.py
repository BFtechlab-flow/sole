"""The wall-clock budget stops training cleanly and a rerun resumes it."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_time_budget_stops_then_resume_finishes(tmp_path, monkeypatch):
    from src.engine import train_fold
    from tools.make_synthetic_dataset import make_dataset

    monkeypatch.delenv("FILANET_DATA_ROOT", raising=False)
    data = make_dataset(tmp_path / "data", n_train=4, n_test=1, size=128, seed=3)
    cfg = yaml.safe_load((ROOT / "configs/smoke_cpu.yaml").read_text())
    cfg["data"]["root"] = str(data)
    cfg["train"].update({"epochs": 3, "patch_size": 96, "val_interval": 1})
    cfg["inference"].update({"tile": 128, "overlap": 0})
    out = tmp_path / "ck"
    logs = []

    first = train_fold(cfg, 0, out, max_hours=1e-6, log=logs.append)
    assert first["finished"] is False and first["completed_epochs"] == 1
    assert (out / "fold0_last.pt").exists() and (out / "fold0_best.pt").exists()
    assert any("time budget" in line for line in logs)

    second = train_fold(cfg, 0, out, log=logs.append)
    assert second["finished"] is True and second["completed_epochs"] == 3
    epochs = [json.loads(line)["epoch"] for line in (out / "fold0_log.jsonl").read_text().splitlines()]
    assert epochs == [1, 2, 3]
