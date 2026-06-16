#!/usr/bin/env python3
"""Phase 7 — vectorize approved raster glyphs to SVG with potrace.

For each data/crops/approved/<NN_letter>.png:
  clean (denoise + binarize) -> upscale -> potrace --svg -> data/crops/approved/svg/<NN_letter>.svg

Run in the venv (uses potrace CLI). Then build_font.py imports the SVGs.

    python src/vectorize.py
"""
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
APPROVED = ROOT / "data" / "crops" / "approved"
SVG_DIR = APPROVED / "svg"

# potrace tuning (see config.yaml notes): smooth curves, drop specks, keep counters.
TURDSIZE = 4        # suppress speckles up to N px
ALPHAMAX = 1.0      # corner smoothness
OPTTOLERANCE = 0.2


def clean(png: Path) -> np.ndarray:
    g = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
    # denoise then Otsu -> ink=255 on 0 (potrace traces the black=set pixels in PBM)
    g = cv2.medianBlur(g, 3)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # upscale 4x so potrace yields smooth outlines, light close to seal pinholes
    b = cv2.resize(b, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, b = cv2.threshold(b, 127, 255, cv2.THRESH_BINARY)
    return b


def main():
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    pngs = sorted(APPROVED.glob("*.png"))
    if not pngs:
        raise SystemExit(f"No approved glyphs in {APPROVED} — run pick_best.py first.")
    for png in pngs:
        ink = clean(png)
        with tempfile.NamedTemporaryFile(suffix=".pbm", delete=False) as tf:
            pbm = Path(tf.name)
        cv2.imwrite(str(pbm), ink)
        svg = SVG_DIR / f"{png.stem}.svg"
        subprocess.run(
            ["potrace", str(pbm), "-s", "-o", str(svg),
             "--turdsize", str(TURDSIZE), "--alphamax", str(ALPHAMAX),
             "--opttolerance", str(OPTTOLERANCE)],
            check=True)
        pbm.unlink()
        print(f"  {png.stem} -> {svg.name}")
    print(f"\nvectorized {len(pngs)} glyphs -> {SVG_DIR}")


if __name__ == "__main__":
    main()
