#!/usr/bin/env python3
"""Phase 8 — assemble the TTF from vectorized glyphs.  Run with FontForge:

    fontforge -script src/build_font.py

Imports data/crops/approved/svg/<NN_letter>.svg into the Hebrew Unicode slots, normalizes
each glyph's size/baseline/side-bearings, and writes output/MyHandwriting.ttf.

Metrics here are a first pass (the make-or-break step): uniform body height, baseline
alignment, below-baseline handling for finals + ק, even side bearings. Tune and re-run.
"""
import os
import sys

import fontforge
import psMat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SVG_DIR = os.path.join(ROOT, "data", "crops", "approved", "svg")
OUT = os.path.join(ROOT, "output", "MyHandwriting.ttf")

ALPHABET = list("אבגדהוזחטיכלמנסעפצקרשת") + list("ךםןףץ")
INDEX = {c: i for i, c in enumerate(ALPHABET)}

EM, ASCENT, DESCENT = 1000, 800, 200
BODY = 600          # target glyph body height (units)
SB = 70             # side bearing each side
DESCENDERS = set("ךןףץק")   # extend below baseline
DROP = 180          # how far descenders dip below baseline


def main():
    font = fontforge.font()
    font.encoding = "UnicodeFull"
    font.em = EM
    font.ascent = ASCENT
    font.descent = DESCENT
    font.familyname = "MyHandwriting"
    font.fontname = "MyHandwriting-Regular"
    font.fullname = "MyHandwriting Regular"

    built, missing = [], []
    for letter in ALPHABET:
        svg = os.path.join(SVG_DIR, "%02d_%s.svg" % (INDEX[letter], letter))
        if not os.path.exists(svg):
            missing.append(letter)
            continue
        g = font.createChar(ord(letter))
        g.importOutlines(svg)
        bb = g.boundingBox()                      # (xmin, ymin, xmax, ymax)
        h = bb[3] - bb[1]
        if h <= 0:
            missing.append(letter)
            continue
        # scale to uniform body height
        g.transform(psMat.scale(BODY / h))
        bb = g.boundingBox()
        # place on baseline (descenders dip below)
        dy = -bb[1] - (DROP if letter in DESCENDERS else 0)
        g.transform(psMat.translate(SB - bb[0], dy))
        bb = g.boundingBox()
        g.width = int(bb[2] + SB)
        g.removeOverlap()
        g.correctDirection()
        built.append(letter)

    font.generate(OUT)
    sys.stderr.write("\nbuilt %d/27 glyphs -> %s\n" % (len(built), OUT))
    if missing:
        sys.stderr.write("MISSING glyphs: %s\n" % " ".join(missing))


main()
