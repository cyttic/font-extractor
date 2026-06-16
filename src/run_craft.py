#!/usr/bin/env python3
"""Run pretrained CRAFT locally to find character (and word) coordinates on real lines.

Uses the official clovaai/CRAFT-pytorch model + utils (in third_party/) with the verified
craft_mlt_25k.pth weights. Writes two overlays per image to output/overlays/:
  craft_word_<name>.png  — word-level boxes (standard linking)
  craft_char_<name>.png  — character-level boxes (linking suppressed)

    python src/run_craft.py <img.png> [<img2.png> ...]
"""
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
CRAFT_DIR = ROOT / "third_party" / "CRAFT-pytorch"
sys.path.insert(0, str(CRAFT_DIR))

import craft_utils
import imgproc
from craft import CRAFT

OVER = ROOT / "output" / "overlays"
OVER.mkdir(parents=True, exist_ok=True)

CANVAS_SIZE = 1280
MAG_RATIO = 3.0          # line crops are short; magnify so CRAFT sees the text big enough


def copy_state(state):
    if list(state.keys())[0].startswith("module"):
        return OrderedDict((k[7:], v) for k, v in state.items())
    return state


def load_net(name="craft_mlt_25k.pth"):
    """Load a CRAFT model by weights filename under models/ (default = stock general model)."""
    weights = ROOT / "models" / name
    net = CRAFT()
    net.load_state_dict(copy_state(torch.load(weights, map_location="cpu")))
    net.eval()
    print(f"[craft] loaded {weights.name}")
    return net


def score_maps(net, image, mag_ratio=MAG_RATIO):
    """One network forward pass -> (score_text, score_link, ratio)."""
    img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(
        image, CANVAS_SIZE, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        y, _ = net(x)
    return y[0, :, :, 0].cpu().numpy(), y[0, :, :, 1].cpu().numpy(), 1 / target_ratio


def boxes_from_scores(score_text, score_link, ratio, text_threshold, link_threshold, low_text):
    boxes, _ = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, False)
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio, ratio)
    return [b for b in boxes if b is not None]


def detect(net, image, text_threshold, link_threshold, low_text):
    st, sl, ratio = score_maps(net, image)
    return boxes_from_scores(st, sl, ratio, text_threshold, link_threshold, low_text)


def draw(image, boxes, color):
    vis = image.copy()
    for b in boxes:
        poly = np.array(b).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [poly], True, color, 2)
    return vis


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    net = load_net()
    for path in sys.argv[1:]:
        path = Path(path)
        image = imgproc.loadImage(str(path))   # RGB uint8
        # word-level: normal linking
        words = detect(net, image, text_threshold=0.7, link_threshold=0.4, low_text=0.4)
        # char-level: suppress linking so each character stays separate
        chars = detect(net, image, text_threshold=0.6, link_threshold=0.95, low_text=0.35)

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(OVER / f"craft_word_{path.stem}.png"), draw(bgr, words, (0, 140, 255)))
        cv2.imwrite(str(OVER / f"craft_char_{path.stem}.png"), draw(bgr, chars, (0, 0, 220)))
        print(f"{path.name}: {len(words)} word-boxes, {len(chars)} char-boxes "
              f"-> output/overlays/craft_[word|char]_{path.stem}.png")


if __name__ == "__main__":
    main()
