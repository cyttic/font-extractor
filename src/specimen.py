#!/usr/bin/env python3
"""Render a specimen image from the built TTF (all 27 letters + a sample sentence).

    python src/specimen.py   ->   output/specimen.png
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
TTF = ROOT / "output" / "MyHandwriting.ttf"
OUT = ROOT / "output" / "specimen.png"

ALPHABET = "א ב ג ד ה ו ז ח ט י כ ל מ נ ס ע פ צ ק ר ש ת   ך ם ן ף ץ"
SENTENCE = "שלום עולם זה הכתב שלי"   # "hello world, this is my handwriting"


def main():
    if not TTF.exists():
        raise SystemExit(f"{TTF} not found — build the font first.")
    img = Image.new("RGB", (1400, 360), "white")
    d = ImageDraw.Draw(img)
    big = ImageFont.truetype(str(TTF), 90)
    small = ImageFont.truetype(str(TTF), 70)
    d.text((40, 30), ALPHABET, fill="black", font=big)
    d.text((40, 180), SENTENCE, fill="black", font=small)
    img.save(OUT)
    print("specimen ->", OUT)


if __name__ == "__main__":
    main()
