import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from src.data import build_input_channels
from src.gong import discover_halpha_urls, download_url, load_halpha_uint8


def panel(title, arr, size=420):
    if arr.ndim == 2:
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).astype(np.uint8)
        arr = np.repeat(arr[..., None], 3, axis=2)
    im = Image.fromarray(arr).resize((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size + 42), "white")
    canvas.paste(im, (0, 42))
    ImageDraw.Draw(canvas).text((10, 12), title, fill="black")
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="Optional direct NOAA GONG .fits.fz URL")
    ap.add_argument("--output-dir", default="artifacts/real_sun")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    url = args.url or discover_halpha_urls(limit=1)[0]
    raw_path = out / "gong_real_halpha.fits.fz"
    image_path = out / "gong_real_halpha.jpg"
    download_url(url, raw_path)

    gray = load_halpha_uint8(raw_path)
    if gray.shape[0] < 512 or gray.shape[1] < 512:
        raise RuntimeError(f"unexpectedly small GONG H-alpha frame: {gray.shape}")
    if not cv2.imwrite(str(image_path), gray):
        raise RuntimeError("failed to save GONG preview image")
    x = build_input_channels(gray)

    panels = [
        panel("REAL NOAA/GONG H-alpha", gray),
        panel("raw normalized", x[0]),
        panel("CLAHE", x[1]),
        panel("ridge", x[2]),
        panel("radial coordinate", x[3]),
    ]
    w = panels[0].width
    h = panels[0].height
    sheet = Image.new("RGB", (w * len(panels), h), "white")
    for i, p in enumerate(panels):
        sheet.paste(p, (i * w, 0))
    sheet.save(out / "real_sun_preprocessing.png")

    np.savez_compressed(out / "real_sun_channels.npz", image=x.astype(np.float32), source_url=np.array(url))
    (out / "source_url.txt").write_text(url + "\n", encoding="utf-8")
    print(f"source={url}")
    print(f"downloaded={raw_path} shape={gray.shape} min={gray.min()} max={gray.max()}")
    print(out / "real_sun_preprocessing.png")


if __name__ == "__main__":
    main()
