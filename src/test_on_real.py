#!/usr/bin/env python3
"""Quick reality check: run the trained classifier on real handwriting line images.

Segments a line into characters (connected components, RTL order), classifies each with
models/classifier.pth, and writes an overlay (predicted letter over each box). Prints the
predicted string next to the ground-truth .gt.txt so we can eyeball synth->real transfer.

    python src/test_on_real.py <line.png> [<line2.png> ...]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classifier import build_model

ROOT = Path(__file__).resolve().parent.parent
OVER = ROOT / "output" / "overlays"
OVER.mkdir(parents=True, exist_ok=True)

# a font that can draw Hebrew labels on the overlay
HEB_FONT = next(iter(sorted(Path("/mnt/ssd2/cyttic/projects/fontsVisualizer/fonts").glob("*.ttf"))), None)


def load_model():
    ckpt = torch.load(ROOT / "models" / "classifier.pth", map_location="cpu")
    model = build_model(len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["classes"], ckpt["img"]


def square_fit(img, size, pad_ratio=0.18, bg=255):
    h, w = img.shape
    pad = int(max(h, w) * pad_ratio)
    side = max(h, w) + 2 * pad
    canvas = np.full((side, side), bg, np.uint8)
    canvas[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = img
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def segment(gray):
    """Connected components -> list of (x, y, w, h) boxes, ordered RTL."""
    _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(binv, connectivity=8)
    H = gray.shape[0]
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 0.0008 * gray.size or h < 0.15 * H:   # drop specks
            continue
        boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[0] + b[2] / 2, reverse=True)   # right-to-left
    return boxes


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    model, classes, img_sz = load_model()
    label_font = ImageFont.truetype(str(HEB_FONT), 40) if HEB_FONT else None

    for path in sys.argv[1:]:
        path = Path(path)
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print("skip (cannot read):", path); continue
        boxes = segment(gray)

        crops = []
        for (x, y, w, h) in boxes:
            crops.append(square_fit(gray[y:y + h, x:x + w], img_sz))
        if not crops:
            print("no components:", path); continue
        batch = torch.from_numpy(np.stack(crops)).float().div_(255).unsqueeze(1)
        with torch.no_grad():
            preds = model(batch).argmax(1).tolist()
        pred_str = "".join(classes[p] for p in preds)

        gt_path = path.with_suffix("").with_suffix(".gt.txt")
        gt = gt_path.read_text().strip() if gt_path.exists() else "(no GT)"

        print(f"\n{path.name}")
        print(f"  GT  : {gt}")
        print(f"  PRED: {pred_str}   ({len(boxes)} components)")

        # overlay
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        pil = Image.fromarray(rgb)
        d = ImageDraw.Draw(pil)
        for (x, y, w, h), p in zip(boxes, preds):
            d.rectangle([x, y, x + w, y + h], outline=(220, 0, 0), width=2)
            if label_font:
                d.text((x, max(0, y - 42)), classes[p], fill=(0, 0, 220), font=label_font)
        out = OVER / f"pred_{path.stem}.png"
        pil.save(out)
        print(f"  overlay -> {out}")


if __name__ == "__main__":
    main()
