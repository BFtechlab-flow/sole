import argparse
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from src.data import build_input_channels


def download(url, path, retries=4):
    headers = {"User-Agent": "FILA-Net/1.0 research demo"}
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read()
            path.write_bytes(data)
            return
        except Exception as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to download {url}: {last}")


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
    ap.add_argument(
        "--url",
        default="https://gong2.nso.edu/ftp/HA/has/201112/20111216/20111216103234Ch.jpg",
    )
    ap.add_argument("--output-dir", default="artifacts/real_sun")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    image_path = out / "gong_real_halpha.jpg"
    download(args.url, image_path)

    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError("downloaded file is not a readable image")
    x = build_input_channels(gray)

    panels = [
        panel("REAL GONG H-alpha", gray),
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

    np.savez_compressed(out / "real_sun_channels.npz", image=x.astype(np.float32))
    print(f"downloaded={image_path} shape={gray.shape} min={gray.min()} max={gray.max()}")
    print(out / "real_sun_preprocessing.png")


if __name__ == "__main__":
    main()
