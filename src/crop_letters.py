"""Split alphabet line-crops into individual letter images by POSITION.

Each input image is a handwritten Hebrew alphabet written in the fixed order
below. We force-align with the CTC aligner (the known sequence is supplied as
the "text" — no recognition happens) and label each segmented letter by its
position. Output: one folder per letter.

Run crop_images.py first to produce the line-crops, then:

    python src/crop_letters.py
    python src/crop_letters.py --src <line_crops_dir> --dst <out_dir> --pad-frac 0.25
"""
import argparse
from pathlib import Path

import cv2

from ctc_align import load_crnn, split_word

# Hebrew alphabet AS WRITTEN ON THE PAGE — 27 glyphs, finals inline.
# (char, ascii folder name); index = position on the line, RIGHT-to-LEFT.
ALPHABET_LINE = [
    ("א", "alef"),  ("ב", "bet"),    ("ג", "gimel"),       ("ד", "dalet"),
    ("ה", "he"),    ("ו", "vav"),    ("ז", "zayin"),       ("ח", "het"),
    ("ט", "tet"),   ("י", "yod"),    ("כ", "kaf"),         ("ך", "kaf_final"),
    ("ל", "lamed"), ("מ", "mem"),    ("ם", "mem_final"),   ("נ", "nun"),
    ("ן", "nun_final"), ("ס", "samekh"), ("ע", "ayin"),    ("פ", "pe"),
    ("ף", "pe_final"),  ("צ", "tsadi"),  ("ץ", "tsadi_final"),
    ("ק", "qof"),   ("ר", "resh"),   ("ש", "shin"),        ("ת", "tav"),
]
TEXT = "".join(c for c, _ in ALPHABET_LINE)   # reading order (RTL), 27 chars

SRC = "/mnt/ssd2/cyttic/datasets/sce_dataset/Dataset_Output/Data/Crops"
DST = "/mnt/ssd2/cyttic/datasets/sce_dataset/Dataset_Output/Data/Letters"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC, help="folder of alphabet line-crops")
    ap.add_argument("--dst", default=DST, help="output root (one subfolder per letter)")
    ap.add_argument("--pad-frac", type=float, default=0.25,
                    help="widen each letter box past its midpoint (overlap allowed)")
    ap.add_argument("--ext", default=".png")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    # pre-create one folder per letter: e.g. 00_alef, 01_bet, …
    letter_dirs = []
    for i, (_, name) in enumerate(ALPHABET_LINE):
        d = dst / f"{i:02d}_{name}"
        d.mkdir(parents=True, exist_ok=True)
        letter_dirs.append(d)

    files = sorted(p for p in src.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    if not files:
        print(f"No images found in {src}")
        return

    print("loading CRNN aligner …")
    net = load_crnn()
    print(f"processing {len(files)} line-crops, {len(ALPHABET_LINE)} letters each")

    done = skipped = 0
    for p in files:
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  skip (unreadable): {p.name}")
            skipped += 1
            continue

        boxes, n = split_word(gray, TEXT, model=net, pad_frac=args.pad_frac)
        if n != len(ALPHABET_LINE):
            print(f"  skip ({n} letters aligned, expected {len(ALPHABET_LINE)}): {p.name}")
            skipped += 1
            continue

        # split_word returns boxes LEFT-to-RIGHT; Hebrew reads RIGHT-to-LEFT,
        # so the rightmost box is letter #0 (א). Sort by x descending.
        boxes_rtl = sorted(boxes, key=lambda b: -b[0])
        for i, (x0, y0, x1, y1) in enumerate(boxes_rtl):
            crop = gray[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            cv2.imwrite(str(letter_dirs[i] / f"{p.stem}{args.ext}"), crop)
        done += 1

    print(f"Done. {done} lines split → {dst}   (skipped {skipped})")


if __name__ == "__main__":
    main()
