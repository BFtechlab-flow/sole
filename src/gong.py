from __future__ import annotations

import re
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np
from astropy.io import fits


H_ALPHA_INDEX = "https://services.swpc.noaa.gov/products/gong/haf/"
_HREF_RE = re.compile(r'href=["\']([^"\']+\.fits\.fz)["\']', re.IGNORECASE)
_FILE_RE = re.compile(r"(?P<stamp>\d{14})(?P<site>[A-Za-z])h\.fits\.fz$", re.IGNORECASE)


def parse_halpha_index(html: str, index_url: str = H_ALPHA_INDEX, limit: int | None = None) -> list[str]:
    names = sorted(set(_HREF_RE.findall(html)))
    urls = [urljoin(index_url, name) for name in names]
    if limit is not None:
        urls = urls[-max(0, int(limit)) :]
    return urls


def halpha_metadata(url: str) -> dict:
    name = Path(urlparse(url).path).name
    match = _FILE_RE.search(name)
    return {
        "url": url,
        "name": name,
        "timestamp": match.group("stamp") if match else None,
        "site": match.group("site").upper() if match else None,
    }


def _evenly_spaced(items: list[str], count: int) -> list[str]:
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return list(items)
    indices = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[int(i)] for i in indices]


def select_diverse_halpha_urls(urls: list[str], count: int) -> list[str]:
    """Deterministically spread samples across observing sites and the index time span."""
    urls = sorted(set(urls))
    count = min(max(0, int(count)), len(urls))
    if count == 0:
        return []

    by_site: dict[str, list[str]] = defaultdict(list)
    unknown = []
    for url in urls:
        meta = halpha_metadata(url)
        if meta["site"]:
            by_site[meta["site"]].append(url)
        else:
            unknown.append(url)

    if not by_site:
        return _evenly_spaced(urls, count)

    sites = sorted(by_site)
    quota = {site: count // len(sites) for site in sites}
    for site in sites[: count % len(sites)]:
        quota[site] += 1

    chosen = []
    for site in sites:
        chosen.extend(_evenly_spaced(by_site[site], min(quota[site], len(by_site[site]))))

    chosen_set = set(chosen)
    if len(chosen) < count:
        remaining = [u for u in urls if u not in chosen_set]
        chosen.extend(_evenly_spaced(remaining, count - len(chosen)))
    return sorted(set(chosen))[:count]


def _read_url(url: str, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "FILA-Net/2.1 solar research"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def discover_halpha_urls(index_url: str = H_ALPHA_INDEX, limit: int | None = 1, retries: int = 3) -> list[str]:
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
