#!/usr/bin/env python3
"""Lexicon-constrained word decoding — turn fuzzy classifier guesses into real words.

For each CRAFT-segmented word, the classifier gives a probability distribution per letter
box. We beam-search candidate spellings under two constraints:
  1. Final-form position rule (free, deterministic): finals ך ם ן ף ץ only at word end;
     base כ מ נ פ צ never at word end.
  2. Hebrew lexicon (wordfreq) with prefix-stripping (ו/ה/ב/כ/ל/מ/ש ...).
Pick the highest-frequency real word among valid candidates. DictaLM 2.0 is the optional
context-aware tie-breaker when the lexicon can't decide.

Eval mode measures decoding vs raw classifier-argmax against the GT on real lines:

    python src/decode.py --limit 300            # measure lift over argmax
    python src/decode.py --limit 300 --use-lm   # add DictaLM fallback
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from wordfreq import word_frequency

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from run_craft import load_net, score_maps, boxes_from_scores
from classifier import build_model

DATA = "/mnt/ssd2/cyttic/datasets/dataset_matan/cleaned_dataset/train_pairs"
ALPHABET = list("אבגדהוזחטיכלמנסעפצקרשת") + list("ךםןףץ")
CLS = {c: i for i, c in enumerate(ALPHABET)}
FINALS = set("ךםןףץ")
HAS_FINAL = set("כמנפצ")          # base letters that MUST become final at word end
PREFIXES = ["ו", "ה", "ב", "כ", "ל", "מ", "ש", "וה", "שה", "כש", "ול", "וב", "ומ", "לה", "מה"]
FREQ_MIN = 1e-6                   # above the wordfreq noise floor => a real word


# ─── lexicon ─────────────────────────────────────────────────────────────────
def is_word(w: str) -> bool:
    if len(w) < 2:
        return word_frequency(w, "he") >= 1e-4
    if word_frequency(w, "he") >= FREQ_MIN:
        return True
    for p in PREFIXES:                       # strip a glued prefix and re-check
        if w.startswith(p) and len(w) - len(p) >= 2 and word_frequency(w[len(p):], "he") >= FREQ_MIN:
            return True
    return False


def freq(w: str) -> float:
    return word_frequency(w, "he")


# ─── decoder ─────────────────────────────────────────────────────────────────
def beam_candidates(probs, topk=5, beam=40):
    """Beam search over per-position class probs, applying the final-form rule.

    Returns [(word, total_logprob)] sorted by classifier confidence (desc).
    """
    L = len(probs)
    beams = [([], 0.0)]
    for i, p in enumerate(probs):
        last = i == L - 1
        order = p.argsort()[::-1][:topk]
        nxt = []
        for letters, lp in beams:
            for j in order:
                ch = ALPHABET[j]
                if ch in FINALS and not last:        # final form mid-word -> illegal
                    continue
                if last and ch in HAS_FINAL:          # base form at word end -> illegal
                    continue
                nxt.append((letters + [ch], lp + math.log(p[j] + 1e-9)))
        nxt.sort(key=lambda t: -t[1])
        beams = nxt[:beam]
    return [("".join(l), lp) for l, lp in beams]


# Only override the classifier's reading with a lexicon word when that word is within
# this per-letter avg-logprob margin of the top reading — i.e. the classifier was unsure.
MARGIN = 0.6


def decode_word(probs, lm=None, context="", topk=5, beam=40):
    cands = beam_candidates(probs, topk, beam)
    if not cands:
        return "".join(ALPHABET[p.argmax()] for p in probs)
    best, best_lp = cands[0]
    if is_word(best):                       # confident reading is already a word -> trust it
        return best
    L = len(best)
    # lexicon repair: first valid candidate close enough in classifier confidence
    valid_near = [(w, lp) for w, lp in cands if is_word(w) and (best_lp - lp) / L <= MARGIN]
    if valid_near:
        return max(valid_near, key=lambda t: t[1])[0]   # highest-confidence valid word
    if lm is not None:
        return lm.best_of([w for w, _ in cands[:8]], context=context)[0]
    return best                              # trust classifier (e.g. a name not in lexicon)


# ─── eval against GT ─────────────────────────────────────────────────────────
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
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--use-lm", action="store_true", help="use DictaLM as fallback tie-breaker")
    args = ap.parse_args()

    net = load_net()
    ckpt = torch.load(ROOT / "models" / "classifier.pth", map_location="cpu")
    clf = build_model(len(ckpt["classes"])); clf.load_state_dict(ckpt["state_dict"]); clf.eval()
    img_sz = ckpt["img"]
    lm = None
    if args.use_lm:
        from dictalm_score import DictaLM
        lm = DictaLM()

    images = sorted(Path(args.data).glob("*.png"))[: args.limit]
    # counters
    w_tot = w_base = w_dec = 0          # word-level exact match
    c_tot = c_base = c_dec = 0          # letter-level
    for path in images:
        gtp = path.with_suffix("").with_suffix(".gt.txt")
        if not gtp.exists():
            continue
        gt_words = gtp.read_text().strip().split()
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        st, sl, ratio = score_maps(net, rgb)
        words = [bbox(b) for b in boxes_from_scores(st, sl, ratio, 0.7, 0.4, 0.4)]
        chars = [bbox(b) for b in boxes_from_scores(st, sl, ratio, 0.6, 0.95, 0.35)]
        words.sort(key=cx, reverse=True); chars.sort(key=cx, reverse=True)
        if len(words) != len(gt_words):
            continue
        for wb, gw in zip(words, gt_words):
            letters = [c for c in gw if c in CLS]
            if len(letters) != len(gw):
                continue
            cb = sorted([c for c in chars if wb[0] - 3 <= cx(c) <= wb[2] + 3], key=cx, reverse=True)
            if len(cb) != len(letters):
                continue
            tiles = [square_fit(gray[max(0, y0):y1, max(0, x0):x1], img_sz) for x0, y0, x1, y1 in cb]
            if any(t is None for t in tiles):
                continue
            batch = torch.from_numpy(np.stack(tiles)).float().div_(255).unsqueeze(1)
            with torch.no_grad():
                probs = clf(batch).softmax(1).numpy()
            base = "".join(ALPHABET[p.argmax()] for p in probs)
            dec = decode_word(probs, lm=lm)
            gw_clean = "".join(letters)
            w_tot += 1
            w_base += (base == gw_clean)
            w_dec += (dec == gw_clean)
            for a, b, g in zip(base, dec, gw_clean):
                c_tot += 1; c_base += (a == g); c_dec += (b == g)

    print(f"\naligned words evaluated: {w_tot}")
    if w_tot:
        print(f"  WORD accuracy   argmax={w_base/w_tot:.3f}   decoded={w_dec/w_tot:.3f}")
        print(f"  LETTER accuracy argmax={c_base/c_tot:.3f}   decoded={c_dec/c_tot:.3f}")
        print(f"  word-level lift: {(w_dec-w_base)/w_tot*100:+.1f} pts   "
              f"letter-level lift: {(c_dec-c_base)/c_tot*100:+.1f} pts")


if __name__ == "__main__":
    main()
