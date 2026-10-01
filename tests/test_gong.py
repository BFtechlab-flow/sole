from pathlib import Path

import numpy as np
from astropy.io import fits

from src.gong import load_halpha_uint8, parse_halpha_index


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


def test_load_halpha_uint8_reads_compressed_image(tmp_path: Path):
    y, x = np.mgrid[:32, :32]
    data = (x + 2 * y).astype(np.float32)
    path = tmp_path / "sample.fits.fz"
    fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(data=data)]).writeto(path)
    image = load_halpha_uint8(path)
    assert image.shape == data.shape
    assert image.dtype == np.uint8
    assert int(image.max()) > int(image.min())
