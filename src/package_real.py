#!/usr/bin/env python3
"""Package harvested real-handwriting crops into real_crops.zip for Kaggle bootstrapping.

Reads data/crops/<NN_letter>/*.png (GT-labeled real crops from harvest.py) and splits them
by WRITER (so val writers are unseen by training) into:

    real_crops/train/<NN_letter>/*.png
    real_crops/val/<NN_letter>/*.png

Then zips to real_crops.zip (upload to Drive like fonts.zip). The classifier label is the
crop's letter directory; the writer id (w### in the filename) drives the split.

    python src/package_real.py            # ~20% of writers held out for val
"""
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CROPS = ROOT / "data" / "crops"
STAGE = ROOT / "data" / "real_crops"
ZIP = ROOT / "real_crops.zip"
VAL_EVERY = 5                       # ~1 in 5 writers -> val
WRITER_RE = re.compile(r"_(w\d+)_")


def writer_of(name: str) -> str:
    m = WRITER_RE.search(name)
    return m.group(1) if m else "w?"


def split_of(writer: str) -> str:
    return "val" if (hash(writer) % VAL_EVERY == 0) else "train"


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    letter_dirs = sorted([d for d in CROPS.iterdir() if d.is_dir() and d.name[0].isdigit()])
    if not letter_dirs:
        raise SystemExit(f"No crop dirs in {CROPS} — run harvest.py first.")

    counts = {"train": Counter(), "val": Counter()}
    writers = set()
    for d in letter_dirs:
        for png in d.glob("*.png"):
            w = writer_of(png.name)
            writers.add(w)
            sp = split_of(w)
            dst = STAGE / sp / d.name
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(png, dst / png.name)
            counts[sp][d.name] += 1

    # zip
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in STAGE.rglob("*.png"):
            z.write(f, f.relative_to(STAGE.parent))   # arcname: real_crops/<split>/<letter>/..

    n_train = sum(counts["train"].values())
    n_val = sum(counts["val"].values())
    print(f"writers: {len(writers)}  | train crops: {n_train}  val crops: {n_val}")
    print("per-letter (train/val):")
    for d in letter_dirs:
        print(f"  {d.name}:  {counts['train'][d.name]:5d} / {counts['val'][d.name]:4d}")
    print(f"\nzip -> {ZIP}  ({ZIP.stat().st_size/1e6:.1f} MB)")
    print("Upload real_crops.zip to Drive, share 'Anyone with the link', paste link in the notebook.")


if __name__ == "__main__":
    main()
