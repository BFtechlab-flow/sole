"""Executes every code cell of notebooks/filanet_end_to_end.ipynb in SMOKE mode.

The notebook is the reference description of the pipeline required by the
organizers, so it must actually run: this test executes its cells in order
(plain Python, no Jupyter needed) on the synthetic dataset and checks the
artifacts it is supposed to produce.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "filanet_end_to_end.ipynb"


def _code_cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            assert "!pip" not in src and "%%" not in src, "notebook must stay executable as plain Python"
            yield src


@pytest.mark.slow
def test_notebook_runs_end_to_end_in_smoke_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("FILANET_MODE", "smoke")
    monkeypatch.setenv("FILANET_ARTIFACTS", str(tmp_path / "artifacts"))
    monkeypatch.delenv("FILANET_DATA_ROOT", raising=False)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    cwd = os.getcwd()
    namespace: dict = {"__name__": "__notebook__"}
    try:
        os.chdir(ROOT)
        for i, src in enumerate(_code_cells()):
            exec(compile(src, f"{NOTEBOOK.name}[code cell {i}]", "exec"), namespace)  # noqa: S102
    finally:
        os.chdir(cwd)

    art = tmp_path / "artifacts"
    for name in ("environment.txt", "data_audit.json", "oof_metrics.json", "oof_metrics_tuned.json",
                 "postprocess_search.json", "ablations/manifest.json", "submission.csv",
                 "submission.verification.json", "oof_export/oof_gt.csv", "oof_export/oof_pred.csv"):
        assert (art / name).exists(), name
    for fig in ("inputs_and_annotations.png", "training_curves.png", "oof_rubric_panels.png", "test_predictions.png"):
        assert (art / "figures" / fig).exists(), fig
    report = json.loads((art / "submission.verification.json").read_text())
    assert report["ok"], report["errors"]
    text = (art / "submission.csv").read_text()
    assert text.startswith("filament_id,segmentation_rle\n") and '"' not in text
    assert namespace["MODE"] == "smoke"
