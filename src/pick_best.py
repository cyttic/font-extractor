#!/usr/bin/env python3
"""Phase 6 (auto-draft) — pick the best harvested crop per letter into approved/.

Reads data/crops/<NN_letter>/ (filenames: <tag>_<conf>_<src>_<i>.png, tag 'ok' = classifier
agreed) and copies the single best crop per letter to data/crops/approved/<NN_letter>.png.
Preference: classifier-agreed first, then highest confidence.

This is a draft selection — swap any letter by replacing its file in approved/ by hand,
or build the Streamlit review UI later for clicking. Run before vectorize.py.
"""
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CROPS = ROOT / "data" / "crops"
APPROVED = CROPS / "approved"
ALPHABET = list("אבגדהוזחטיכלמנסעפצקרשת") + list("ךםןףץ")
CLS = {c: i for i, c in enumerate(ALPHABET)}


def score(name):
    # name: <tag>_<conf>_<src>_<i>.png  -> (agreed?, conf)
    m = re.match(r"(ok|x)_([0-9.]+)_", name)
    if not m:
        return (0, 0.0)
    return (1 if m.group(1) == "ok" else 0, float(m.group(2)))


def main():
    APPROVED.mkdir(parents=True, exist_ok=True)
    picked, missing = [], []
    for letter in ALPHABET:
        d = CROPS / f"{CLS[letter]:02d}_{letter}"
        files = list(d.glob("*.png")) if d.exists() else []
        if not files:
            missing.append(letter)
            continue
        best = max(files, key=lambda f: score(f.name))
        dst = APPROVED / f"{CLS[letter]:02d}_{letter}.png"
        shutil.copy(best, dst)
        picked.append((letter, best.name))
    for letter, src in picked:
        print(f"  {letter}  <- {src}")
    print(f"\napproved {len(picked)}/27 -> {APPROVED}")
    if missing:
        print("MISSING (no crop harvested):", " ".join(missing))


if __name__ == "__main__":
    main()
