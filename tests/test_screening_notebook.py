"""Executes every code cell of notebooks/screening_fold.ipynb in SMOKE mode (synthetic data, CPU)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "screening_fold.ipynb"


@pytest.mark.slow
def test_screening_notebook_runs_in_smoke_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("FILANET_MODE", "smoke")
    monkeypatch.setenv("FILANET_ARTIFACTS", str(tmp_path / "artifacts"))
    monkeypatch.delenv("FILANET_DATA_ROOT", raising=False)
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cwd = os.getcwd()
    namespace: dict = {"__name__": "__notebook__"}
    try:
        os.chdir(ROOT)
        for i, cell in enumerate(c for c in nb["cells"] if c["cell_type"] == "code"):
            src = "".join(cell["source"])
            assert "!pip" not in src and "%%" not in src
            exec(compile(src, f"{NOTEBOOK.name}[code cell {i}]", "exec"), namespace)  # noqa: S102
    finally:
        os.chdir(cwd)
    rows = json.loads((tmp_path / "artifacts" / "screening" / "screening_results.json").read_text())
    models = {r["model"] for r in rows}
    assert {"baseline", "seed2", "baseline+seed2", "convnext_small", "baseline+convnext_small"} <= models
    assert all("delta" in r for r in rows if r["model"] != "baseline")
    # the screening checkpoints are never taken as fold checkpoints by the main notebook
    from src.sessions import find_checkpoint_dirs

    assert find_checkpoint_dirs([tmp_path]) == []
