#!/usr/bin/env python3
"""Harvest labeled letter glyphs from real handwriting using CRAFT boxes + GT text.

For each line image (with its .gt.txt):
  1. One CRAFT forward pass -> word boxes (normal linking) and char boxes (linking suppressed).
  2. Split GT into words; keep lines where #word-boxes == #GT-words (clean alignment).
  3. Per word, assign char boxes by horizontal position, sorted right-to-left (Hebrew).
     Keep words where #char-boxes == #GT-letters -> each char box gets its GT label for free.
  4. Score each crop with the trained classifier (agreement = quality), save to data/crops/<letter>/.

Then build a per-letter gallery (top crops by confidence) for human review.

    python src/harvest.py --limit 400
    python src/harvest.py --data <dir> --limit 1000
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from run_craft import load_net, score_maps, boxes_from_scores
from classifier import build_model

DATA_DEFAULT = "/mnt/ssd2/cyttic/datasets/dataset_matan/cleaned_dataset/train_pairs"
ALPHABET = list("אבגדהוזחטיכלמנסעפצקרשת") + list("ךםןףץ")
CLS = {c: i for i, c in enumerate(ALPHABET)}
CROPS = ROOT / "data" / "crops"
OVER = ROOT / "output" / "overlays"


def bbox(poly):
    p = np.array(poly)
    return int(p[:, 0].min()), int(p[:, 1].min()), int(p[:, 0].max()), int(p[:, 1].max())


def cx(b):
    return (b[0] + b[2]) / 2


def square_fit(img, size, pad_ratio=0.18, bg=255):
    h, w = img.shape
    if h == 0 or w == 0:
        return None
    pad = int(max(h, w) * pad_ratio)
    side = max(h, w) + 2 * pad
    c = np.full((side, side), bg, np.uint8)
    c[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = img
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--gallery-top", type=int, default=24)
    args = ap.parse_args()

    net = load_net()
    ckpt = torch.load(ROOT / "models" / "classifier.pth", map_location="cpu")
    clf = build_model(len(ckpt["classes"])); clf.load_state_dict(ckpt["state_dict"]); clf.eval()
    img_sz = ckpt["img"]

    images = sorted(Path(args.data).glob("*.png"))[: args.limit]
    print(f"processing {len(images)} lines from {args.data}")

    # buckets: letter -> list of (confidence, crop_uint8, source)
    buckets = defaultdict(list)
    n_lines_aligned = 0
    for k, path in enumerate(images):
        gt_path = path.with_suffix("").with_suffix(".gt.txt")
        if not gt_path.exists():
            continue
        gt_words = gt_path.read_text().strip().split()
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        st, sl, ratio = score_maps(net, rgb)
        word_boxes = [bbox(b) for b in boxes_from_scores(st, sl, ratio, 0.7, 0.4, 0.4)]
        char_boxes = [bbox(b) for b in boxes_from_scores(st, sl, ratio, 0.6, 0.95, 0.35)]
        word_boxes.sort(key=cx, reverse=True)   # right-to-left
        char_boxes.sort(key=cx, reverse=True)

        if len(word_boxes) != len(gt_words):
            continue                            # ambiguous line, skip

        line_crops = []   # (letter, crop64)
        ok = True
        for wb, gword in zip(word_boxes, gt_words):
            letters = [c for c in gword if c in CLS]
            if len(letters) != len(gword):      # punctuation/digits -> skip word
                continue
            chars = sorted([c for c in char_boxes if wb[0] - 3 <= cx(c) <= wb[2] + 3], key=cx, reverse=True)
            if len(chars) != len(letters):
                continue                        # word doesn't cleanly segment -> skip word
            for cb, letter in zip(chars, letters):
                x0, y0, x1, y1 = cb
                crop = gray[max(0, y0):y1, max(0, x0):x1]
                tile = square_fit(crop, img_sz)
                if tile is not None:
                    line_crops.append((letter, tile))
        if not line_crops:
            continue
        n_lines_aligned += 1

        # classifier scores (batch) -> confidence that crop matches its GT letter
        batch = torch.from_numpy(np.stack([t for _, t in line_crops])).float().div_(255).unsqueeze(1)
        with torch.no_grad():
            probs = clf(batch).softmax(1).numpy()
        for (letter, tile), p in zip(line_crops, probs):
            conf = float(p[CLS[letter]])
            agree = int(p.argmax() == CLS[letter])   # classifier confirms the GT label
            buckets[letter].append((agree, conf, tile, path.stem))

        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(images)} lines  | aligned so far: {n_lines_aligned}")

    # save crops + galleries
    for d in (CROPS, OVER):
        d.mkdir(parents=True, exist_ok=True)
    print("\nper-letter harvest (total | agreed = CRAFT+GT+classifier all match):")
    summary = []
    for letter in ALPHABET:
        # sort: agreed crops first, then by confidence
        items = sorted(buckets[letter], key=lambda t: (-t[0], -t[1]))
        n_agree = sum(a for a, _, _, _ in items)
        ld = CROPS / f"{CLS[letter]:02d}_{letter}"
        ld.mkdir(parents=True, exist_ok=True)
        for i, (agree, conf, tile, src) in enumerate(items):
            tag = "ok" if agree else "x"
            cv2.imwrite(str(ld / f"{tag}_{conf:.3f}_{src}_{i}.png"), tile)
        summary.append((letter, len(items), n_agree))
        # gallery montage of top-N (agreed first)
        top = items[: args.gallery_top]
        if top:
            cols = 8
            rows = (len(top) + cols - 1) // cols
            sz = img_sz
            sheet = np.full((rows * sz, cols * sz), 255, np.uint8)
            for i, (agree, conf, tile, src) in enumerate(top):
                r, c = divmod(i, cols)
                sheet[r * sz:(r + 1) * sz, c * sz:(c + 1) * sz] = tile
            cv2.imwrite(str(OVER / f"gallery_{CLS[letter]:02d}_{letter}.png"), sheet)

    for letter, n, n_agree in summary:
        bar = "#" * min(n_agree, 40)
        print(f"  {letter}  total={n:4d}  agreed={n_agree:4d}  {bar}")
    missing = [l for l, n, _ in summary if n == 0]
    no_agree = [l for l, _, a in summary if a == 0]
    print(f"\nlines aligned: {n_lines_aligned}/{len(images)}")
    print(f"letters with >=1 candidate: {27 - len(missing)}/27")
    print(f"letters with >=1 AGREED (clean) candidate: {27 - len(no_agree)}/27")
    if missing:
        print("MISSING entirely:", " ".join(missing))
    if no_agree:
        print("No agreed crop:", " ".join(no_agree))
    print(f"crops -> {CROPS}\ngalleries -> {OVER}/gallery_*.png")


if __name__ == "__main__":
    main()
