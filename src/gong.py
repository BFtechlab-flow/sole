from __future__ import annotations

import re
import time
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
from astropy.io import fits


H_ALPHA_INDEX = "https://services.swpc.noaa.gov/products/gong/haf/"
_HREF_RE = re.compile(r'href=["\']([^"\']+\.fits\.fz)["\']', re.IGNORECASE)


def parse_halpha_index(html: str, index_url: str = H_ALPHA_INDEX, limit: int | None = None) -> list[str]:
    names = sorted(set(_HREF_RE.findall(html)))
    urls = [urljoin(index_url, name) for name in names]
    if limit is not None:
        urls = urls[-max(0, int(limit)) :]
    return urls


def _read_url(url: str, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "FILA-Net/2.0 solar research"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def discover_halpha_urls(index_url: str = H_ALPHA_INDEX, limit: int = 1, retries: int = 3) -> list[str]:
    last = None
    for attempt in range(retries):
        try:
            html = _read_url(index_url).decode("utf-8", errors="ignore")
            urls = parse_halpha_index(html, index_url=index_url, limit=limit)
            if urls:
                return urls
            raise RuntimeError("NOAA GONG H-alpha index contained no .fits.fz files")
        except Exception as exc:
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to discover NOAA GONG H-alpha files: {last}")


def download_url(url: str, path: str | Path, retries: int = 3, timeout: float = 30.0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(retries):
        try:
            data = _read_url(url, timeout=timeout)
            if len(data) < 1024:
                raise RuntimeError(f"download too small: {len(data)} bytes")
            path.write_bytes(data)
            return path
        except Exception as exc:
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {url}: {last}")


def load_halpha_uint8(path: str | Path, low_percentile: float = 0.5, high_percentile: float = 99.5) -> np.ndarray:
    path = Path(path)
    with fits.open(path, memmap=False) as hdul:
        data = None
        for hdu in hdul:
            arr = getattr(hdu, "data", None)
            if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                data = np.squeeze(arr).astype(np.float32)
                break
    if data is None or data.ndim != 2:
        raise RuntimeError(f"no 2-D image found in {path}")

    finite = np.isfinite(data)
    if not finite.any():
        raise RuntimeError(f"no finite pixels found in {path}")
    values = data[finite]
    lo, hi = np.percentile(values, [low_percentile, high_percentile])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return np.zeros_like(data, dtype=np.uint8)
    scaled = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    scaled[~finite] = 0.0
    return np.round(scaled * 255.0).astype(np.uint8)
