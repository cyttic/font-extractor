#!/usr/bin/env python3
"""Build a CRAFT fine-tune package from real handwriting (dataset_matan), weak-supervised.

For each line: stock CRAFT finds words; we align them to the GT words (count must match);
for each word, connected components give pseudo per-letter boxes — kept ONLY when the CC
count equals the GT word's letter count (trustworthy targets). Each kept word is normalized
onto a fixed 64x256 white canvas with its letter boxes, and saved.

Output: craft_train/imgs/*.png + craft_train/boxes.json, zipped to craft_train.zip
(upload to Drive, then run notebooks/kaggle_craft_finetune_real.ipynb).

    python src/prep_craft_data.py            # full dataset
    python src/prep_craft_data.py --limit 300
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from run_craft import load_net, score_maps, boxes_from_scores

DATA = "/mnt/ssd2/cyttic/datasets/dataset_matan/cleaned_dataset/train_pairs"
ALPHASET = set("אבגדהוזחטיכלמנסעפצקרשת" + "ךםןףץ")
TH, TW, PAD = 64, 256, 4          # fixed training canvas (height, width, vertical pad)
OUT = ROOT / "data" / "craft_train"
ZIP = ROOT / "craft_train.zip"


def bbox(poly):
    p = np.array(poly)
    return int(p[:, 0].min()), int(p[:, 1].min()), int(p[:, 0].max()), int(p[:, 1].max())


def cx(b): return (b[0] + b[2]) / 2


def cc_boxes(gray_word):
    inv = 255 - gray_word
    _, b = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    nlab, _, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    H = gray_word.shape[0]
    out = []
    for i in range(1, nlab):
        x, y, w, h, area = stats[i]
        if area > 0.001 * gray_word.size and h > 0.2 * H:
            out.append((x, y, x + w, y + h))
    return sorted(out, key=cx, reverse=True)        # right-to-left


def normalize_word(gray_word, boxes):
    """Place a word crop on a fixed THxTW white canvas; scale its boxes to match."""
    h, w = gray_word.shape
    s = (TH - 2 * PAD) / h
    if w * s > TW:
        s = TW / w
    rw = cv2.resize(gray_word, (max(1, int(w * s)), max(1, int(h * s))))
    canvas = np.full((TH, TW), 255, np.uint8)
    y0 = (TH - rw.shape[0]) // 2
    canvas[y0:y0 + rw.shape[0], 0:rw.shape[1]] = rw
    nb = [[int(a * s), int(b * s) + y0, int(c * s), int(d * s) + y0] for (a, b, c, d) in boxes]
    return canvas, nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--limit", type=int, default=6000)
    args = ap.parse_args()

    net = load_net()                              # stock CRAFT for word detection
    (OUT / "imgs").mkdir(parents=True, exist_ok=True)
    images = sorted(Path(args.data).glob("*.png"))[: args.limit]
    boxes_map = {}
    idx = aligned_lines = 0
    for k, path in enumerate(images):
        gtp = path.with_suffix("").with_suffix(".gt.txt")
        if not gtp.exists():
            continue
        gt_words = gtp.read_text().strip().split()
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        st, sl, ratio = score_maps(net, rgb)
        wbs = sorted([bbox(b) for b in boxes_from_scores(st, sl, ratio, 0.7, 0.4, 0.4)],
                     key=cx, reverse=True)
        if len(wbs) != len(gt_words):
            continue
        aligned_lines += 1
        for wb, gw in zip(wbs, gt_words):
            letters = [c for c in gw if c in ALPHASET]
            if len(letters) != len(gw) or len(letters) < 2:        # skip punctuation / 1-char
                continue
            crop = gray[max(0, wb[1]):wb[3], max(0, wb[0]):wb[2]]
            if crop.size == 0:
                continue
            cb = cc_boxes(crop)
            if len(cb) != len(letters):                            # trustworthy targets only
                continue
            canvas, nb = normalize_word(crop, cb)
            fn = f"{idx:06d}.png"
            cv2.imwrite(str(OUT / "imgs" / fn), canvas)
            boxes_map[fn] = nb
            idx += 1
        if (k + 1) % 500 == 0:
            print(f"  {k+1}/{len(images)} lines | aligned {aligned_lines} | word samples {idx}")

    (OUT / "boxes.json").write_text(json.dumps(boxes_map))
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(OUT / "boxes.json", "craft_train/boxes.json")
        for f in (OUT / "imgs").glob("*.png"):
            z.write(f, f"craft_train/imgs/{f.name}")
    print(f"\naligned lines: {aligned_lines} | word samples: {idx}")
    print(f"zip -> {ZIP}  ({ZIP.stat().st_size/1e6:.1f} MB)")
    print("Upload craft_train.zip to Drive, share 'Anyone with link', paste link in the notebook.")


if __name__ == "__main__":
    main()
