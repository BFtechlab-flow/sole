"""Publish and retrieve the trained fold checkpoints (reproducibility rule).

The organizers must be able to reproduce the submitted predictions from the
public repository alone. Checkpoints are too large for git, so they are
attached to a GitHub Release and listed, with their SHA-256, in
``weights/manifest.json`` (committed). Anybody can then download and verify
them with one command.

Typical use::

    # 1. after training: write the manifest (URLs = where the files will be published)
    python tools/weights.py manifest \
        --checkpoints checkpoints/fold0_best.pt checkpoints/fold1_best.pt \
                      checkpoints/fold2_best.pt checkpoints/fold3_best.pt checkpoints/fold4_best.pt \
        --base-url https://github.com/BFtechlab-flow/sole/releases/download/v1.0/ \
        --config configs/filanet_tuned.yaml
    # 2. upload the five .pt files to the GitHub release "v1.0" (web UI), commit weights/manifest.json
    # 3. anyone, later:
    python tools/weights.py fetch --dest checkpoints      # downloads + verifies SHA-256
    python tools/weights.py verify --dest checkpoints     # verifies local copies only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_MANIFEST = "weights/manifest.json"


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_summary(path: Path) -> dict:
    """Fold / epoch / validation metrics stored by ``src.engine.train_fold``."""
    try:
        import torch

        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - the manifest is still useful without it
        return {"note": f"could not read checkpoint metadata: {exc}"}
    metrics = ck.get("metrics", {}) or {}
    return {
        "fold": ck.get("fold"),
        "epoch": ck.get("epoch"),
        "schema_version": ck.get("schema_version"),
        "validation_micro_pq": metrics.get("pq"),
        "validation_images": metrics.get("validation_images"),
    }


def build_manifest(checkpoints, base_url: str, config: str | None = None, description: str = "") -> dict:
    if not base_url.endswith("/"):
        base_url += "/"
    files = []
    for ck in checkpoints:
        p = Path(ck)
        if not p.is_file():
            raise FileNotFoundError(p)
        entry = {"name": p.name, "url": base_url + p.name, "sha256": sha256_file(p), "bytes": p.stat().st_size}
        entry.update(_checkpoint_summary(p))
        files.append(entry)
    names = [f["name"] for f in files]
    if len(set(names)) != len(names):
        raise ValueError(f"checkpoint file names must be unique: {names}")
    return {
        "description": description or "FILA-Net fold checkpoints used for the submitted predictions",
        "config": config,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": files,
    }


def _download(url: str, dest: Path, retries: int = 3, timeout: float = 60.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "FILA-Net weights fetcher"})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 20)
            tmp.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001 - retried, then reported
            last = exc
            time.sleep(2 ** attempt)
    if tmp.exists():
        tmp.unlink()
    raise RuntimeError(f"download failed for {url}: {last}")


def verify(manifest: dict, dest) -> list[str]:
    """Return the list of problems (empty when every file is present and intact)."""
    problems = []
    for f in manifest["files"]:
        p = Path(dest) / f["name"]
        if not p.is_file():
            problems.append(f"missing {p}")
        elif sha256_file(p) != f["sha256"]:
            problems.append(f"SHA-256 mismatch for {p}")
    return problems


def fetch(manifest: dict, dest, force: bool = False) -> list[Path]:
    paths = []
    for f in manifest["files"]:
        p = Path(dest) / f["name"]
        if force or not p.is_file() or sha256_file(p) != f["sha256"]:
            print(f"downloading {f['url']} -> {p}", flush=True)
            _download(f["url"], p)
        if sha256_file(p) != f["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch after download: {p}")
        paths.append(p)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest", help="write the manifest of trained checkpoints")
    m.add_argument("--checkpoints", nargs="+", required=True)
    m.add_argument("--base-url", required=True, help="URL prefix where the files are (or will be) published")
    m.add_argument("--config", default=None, help="config used with these weights (e.g. configs/filanet_tuned.yaml)")
    m.add_argument("--description", default="")
    m.add_argument("--output", default=DEFAULT_MANIFEST)
    for name in ("fetch", "verify"):
        s = sub.add_parser(name)
        s.add_argument("--manifest", default=DEFAULT_MANIFEST)
        s.add_argument("--dest", default="checkpoints")
        if name == "fetch":
            s.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "manifest":
        manifest = build_manifest(args.checkpoints, args.base_url, args.config, args.description)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out} ({len(manifest['files'])} files)")
        return 0

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if not manifest.get("files"):
        print(f"{args.manifest} lists no files yet (no trained weights have been published)", file=sys.stderr)
        return 2
    if args.cmd == "fetch":
        for p in fetch(manifest, args.dest, force=args.force):
            print(f"ok {p}")
        return 0
    problems = verify(manifest, args.dest)
    for line in problems:
        print(line, file=sys.stderr)
    print("all checkpoints verified" if not problems else f"{len(problems)} problem(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
