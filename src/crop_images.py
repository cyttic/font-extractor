"""Crop a fixed rectangle from every .tif image in a folder and save the crops.

Default rectangle: x=423, y=1290, size 3840x190.

Usage:
    python src/crop_images.py
    python src/crop_images.py --src <dir> --dst <dir> --x 423 --y 1290 --w 3840 --h 190
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image   # Pillow handles old-style JPEG-compressed TIFFs

SRC = "/mnt/ssd2/cyttic/datasets/sce_dataset/Dataset_Output/Data/Images"
DST = "/mnt/ssd2/cyttic/datasets/sce_dataset/Dataset_Output/Data/Crops"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--x", type=int, default=423)
    ap.add_argument("--y", type=int, default=1290)
    ap.add_argument("--w", type=int, default=3840)
    ap.add_argument("--h", type=int, default=190)
    ap.add_argument("--ext", default=".png",
                    help="output file extension (.png, .jpg, .tif)")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src.iterdir()
                   if p.suffix.lower() in (".tif", ".tiff"))
    if not files:
        print(f"No .tif/.tiff images found in {src}")
        return

    x0, y0, x1, y1 = args.x, args.y, args.x + args.w, args.y + args.h
    done = skipped = 0
    for p in files:
        try:
            img = Image.open(p)
        except Exception as e:
            print(f"  skip (open failed): {p.name}  ({e})")
            skipped += 1
            continue
        W, H = img.size
        if x1 > W or y1 > H:
            print(f"  skip (crop {x1}x{y1} exceeds image {W}x{H}): {p.name}")
            skipped += 1
            continue
        crop = img.crop((x0, y0, x1, y1))
        out = dst / (p.stem + args.ext)
        crop.save(out)
        done += 1

    print(f"Cropped {done} images → {dst}   (skipped {skipped})")


if __name__ == "__main__":
    main()
