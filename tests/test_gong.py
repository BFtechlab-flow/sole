from pathlib import Path

import numpy as np
from astropy.io import fits

from src.gong import (
    halpha_metadata,
    load_halpha_uint8,
    parse_halpha_index,
    select_diverse_halpha_urls,
)


def test_parse_halpha_index_returns_latest_files():
    html = '''
    <a href="20260930031302Uh.fits.fz">a</a>
    <a href="20260930031442Lh.fits.fz">b</a>
    <a href="ignore.txt">x</a>
    <a href="20260930031502Uh.fits.fz">c</a>
    '''
    urls = parse_halpha_index(html, "https://example.test/haf/", limit=2)
    assert urls == [
        "https://example.test/haf/20260930031442Lh.fits.fz",
        "https://example.test/haf/20260930031502Uh.fits.fz",
    ]


def test_diverse_selection_uses_multiple_sites_and_time_span():
    urls = []
    for minute in range(10):
        for site in ("U", "L", "C"):
            urls.append(f"https://example.test/haf/2026100110{minute:02d}02{site}h.fits.fz")
    selected = select_diverse_halpha_urls(urls, 6)
    sites = {halpha_metadata(u)["site"] for u in selected}
    stamps = sorted(halpha_metadata(u)["timestamp"] for u in selected)
    assert len(selected) == 6
    assert sites == {"U", "L", "C"}
    assert stamps[0] < stamps[-1]


def test_load_halpha_uint8_reads_compressed_image(tmp_path: Path):
    y, x = np.mgrid[:32, :32]
    data = (x + 2 * y).astype(np.float32)
    path = tmp_path / "sample.fits.fz"
    fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(data=data)]).writeto(path)
    image = load_halpha_uint8(path)
    assert image.shape == data.shape
    assert image.dtype == np.uint8
    assert int(image.max()) > int(image.min())
