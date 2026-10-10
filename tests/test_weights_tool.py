"""tools/weights.py: manifest -> fetch (file:// URLs) -> SHA-256 verification."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tools.weights import build_manifest, fetch, main, verify


def _fake_checkpoint(path: Path, fold: int):
    torch.save({"schema_version": 4, "model": {"w": torch.ones(3) * fold}, "fold": fold, "epoch": 7,
                "metrics": {"pq": 0.25 + fold / 100, "validation_images": 10}}, path)


def test_manifest_fetch_and_verify(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    for f in range(2):
        _fake_checkpoint(release / f"fold{f}_best.pt", f)
    manifest = build_manifest(sorted(release.glob("*.pt")), release.as_uri(), config="configs/filanet.yaml")
    assert [f["name"] for f in manifest["files"]] == ["fold0_best.pt", "fold1_best.pt"]
    assert manifest["files"][1]["fold"] == 1 and manifest["files"][1]["validation_micro_pq"] == pytest.approx(0.26)

    dest = tmp_path / "checkpoints"
    paths = fetch(manifest, dest)
    assert [p.name for p in paths] == ["fold0_best.pt", "fold1_best.pt"]
    assert verify(manifest, dest) == []

    # a corrupted local copy is detected, and fetch repairs it
    (dest / "fold0_best.pt").write_bytes(b"corrupted")
    assert verify(manifest, dest) == [f"SHA-256 mismatch for {dest / 'fold0_best.pt'}"]
    fetch(manifest, dest)
    assert verify(manifest, dest) == []


def test_cli_refuses_an_empty_manifest(tmp_path):
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({"files": []}))
    assert main(["fetch", "--manifest", str(m), "--dest", str(tmp_path / "ck")]) == 2


def test_repository_manifest_is_valid_json():
    manifest = json.loads((Path(__file__).resolve().parents[1] / "weights" / "manifest.json").read_text())
    assert "files" in manifest
    for f in manifest["files"]:
        assert {"name", "url", "sha256"} <= set(f)
