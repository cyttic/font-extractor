"""Diagnose CRNN/CTC alignment on real dataset word crops (no manual cropping).

Pulls a few random crops from Words_Dataset, looks up their text via the JSON
class map, runs the full alignment, and prints what the model emits.

Usage:
    python src/debug_ctc.py
    python src/debug_ctc.py /path/to/word.png "המילה"   # single custom image
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crnn import IDX2CHAR, BLANK, CHAR2IDX, encode, decode
from ctc_align import load_crnn, _prepare, _viterbi_align, split_word

DATASET = Path("/mnt/ssd2/cyttic/datasets/sce_dataset/Dataset_Output")


def build_class_map(json_dir: Path) -> dict[int, str]:
    class_texts = defaultdict(Counter)
    for jf in json_dir.glob("*.json"):
        try:
            data = json.load(open(jf, encoding="utf-8"))
        except Exception:
            continue
        for s in data.get("shapes", []):
            if s.get("type") == "word" and s.get("word_class") is not None:
                t = s.get("transcript", "").strip()
                if t:
                    class_texts[s["word_class"]][t] += 1
    return {c: ctr.most_common(1)[0][0] for c, ctr in class_texts.items()}


def diagnose(gray: np.ndarray, text: str, model) -> None:
    print("=" * 70)
    print(f"text(RTL)={text!r}  size={gray.shape[1]}x{gray.shape[0]}")

    tensor, feat_w = _prepare(gray)
    with torch.no_grad():
        log_probs = model(tensor)[:, 0, :].numpy()
    T = log_probs.shape[0]
    greedy = log_probs.argmax(1)
    print(f"  T={T}  greedy(RTL)={decode(greedy)[::-1]!r}")

    spikes = [(t, IDX2CHAR.get(int(c), '?')) for t, c in enumerate(greedy) if c != BLANK]
    print(f"  spikes={spikes}")

    labels = encode(text[::-1])
    centers = _viterbi_align(log_probs, labels)
    print(f"  N={len(labels)}  centers={[round(c, 1) for c in centers]}")

    boxes, n = split_word(gray, text, model=model)
    print(f"  box widths={[b[2]-b[0] for b in boxes]}  (img W={gray.shape[1]})")


def main():
    model = load_crnn()

    if len(sys.argv) >= 3:
        gray = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
        diagnose(gray, sys.argv[2], model)
        return

    print("Building class map …")
    cmap = build_class_map(DATASET / "Data" / "json_labels")
    words_dir = DATASET / "Words_Dataset"

    # pick 6 random crops from classes we can label
    picks = []
    cls_dirs = [d for d in words_dir.iterdir() if d.name.isdigit() and int(d.name) in cmap]
    for d in random.sample(cls_dirs, min(6, len(cls_dirs))):
        text = "".join(c for c in cmap[int(d.name)] if c in CHAR2IDX)
        imgs = list(d.glob("*.jpg"))
        if imgs and text:
            picks.append((random.choice(imgs), text))

    for img_path, text in picks:
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is not None:
            diagnose(gray, text, model)


if __name__ == "__main__":
    main()
