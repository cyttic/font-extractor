#!/usr/bin/env python3
"""FastAPI demo — TrOCR-driven letter extraction.

Pipeline:
  preprocess -> CRAFT splits the page into WORDS (reading order, RTL + top-to-bottom)
  -> TrOCR reads each word + confidence
  -> for words with confidence >= CONF_MIN, force-split the word image into exactly
     len(reading) letters via vertical projection valleys, label them from the OCR text.
The classifier is NOT used for results — the labels come from TrOCR, the crops from the split.

    .venv/bin/uvicorn src.app:app --reload --port 8000   ->  http://localhost:8000
    (needs the TrOCR server on :8001 — see webHebrewOCR)
"""
import base64
import sys
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from run_craft import load_net, score_maps, boxes_from_scores
from classifier import build_model
from preprocess import preprocess

ALPHABET = list("אבגדהוזחטיכלמנסעפצקרשת") + list("ךםןףץ")
ALPHASET = set(ALPHABET)
CONF_MIN = 0.90        # only harvest letters from words TrOCR is sure about
LETTER_OK = 0.80       # classifier P(OCR letter) required to accept a segmented letter
                       # (0.95 yields ~9% on this classifier; 0.80 ~55% at high precision)

OCR_URL = "http://127.0.0.1:8001"
OCR_MODEL = "trocr-hebrew-matan-exp7"
OCR_BEAMS = 4

print("loading CRAFT + classifier ...")
NET = load_net()                                   # stock model: robust WORD detection
_heb = ROOT / "models" / "craft_hebrew.pth"
NET_LETTERS = load_net("craft_hebrew.pth") if _heb.exists() else NET   # fine-tuned: LETTER split
_ck = torch.load(ROOT / "models" / "classifier.pth", map_location="cpu")
CLF = build_model(len(_ck["classes"])); CLF.load_state_dict(_ck["state_dict"]); CLF.eval()
IMG = _ck["img"]
CLASSES = _ck.get("classes") or ALPHABET
CLS = {c: i for i, c in enumerate(CLASSES)}
print("ready.")


def classify(tiles):
    batch = torch.from_numpy(np.stack(tiles)).float().div_(255).unsqueeze(1)
    with torch.no_grad():
        return CLF(batch).softmax(1).numpy()

app = FastAPI(title="Hebrew Handwriting → TrOCR → Letters")


# ── OCR client ───────────────────────────────────────────────────────────────
def ocr_available() -> bool:
    try:
        return requests.get(f"{OCR_URL}/health", timeout=2).status_code == 200
    except Exception:
        return False


def ocr_word(gray_crop):
    ok, buf = cv2.imencode(".png", gray_crop)
    try:
        r = requests.post(f"{OCR_URL}/ocr",
                          data={"model": OCR_MODEL, "beams": OCR_BEAMS},
                          files={"file": ("word.png", buf.tobytes(), "image/png")},
                          timeout=30)
        if r.status_code != 200:
            return None
        j = r.json()
        return j.get("text", ""), float(j.get("confidence", 0.0))
    except Exception:
        return None


def conf_color(c: float) -> str:
    return "#2a7" if c >= 0.8 else ("#c80" if c >= 0.5 else "#c00")


# ── geometry helpers ─────────────────────────────────────────────────────────
def bbox(poly):
    p = np.array(poly)
    return int(p[:, 0].min()), int(p[:, 1].min()), int(p[:, 0].max()), int(p[:, 1].max())


def cx(b): return (b[0] + b[2]) / 2
def cy(b): return (b[1] + b[3]) / 2


def center_in(c, w, m=3):
    """True if char box c's center lies within word box w (small margin)."""
    return w[0] - m <= cx(c) <= w[2] + m and w[1] - m <= cy(c) <= w[3] + m


def reading_order(boxes):
    if not boxes:
        return []
    h = float(np.median([b[3] - b[1] for b in boxes]))
    lines, cur, ref = [], [], None
    for b in sorted(boxes, key=cy):
        if ref is None or abs(cy(b) - ref) <= 0.6 * h:
            cur.append(b); ref = cy(b) if ref is None else (ref + cy(b)) / 2
        else:
            lines.append(cur); cur = [b]; ref = cy(b)
    if cur:
        lines.append(cur)
    out = []
    for ln in lines:
        out.extend(sorted(ln, key=cx, reverse=True))   # RTL within a line
    return out


def png_b64(img) -> str:
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def ink_profile(gray_word):
    inv = 255 - gray_word
    _, b = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b, b.sum(axis=0).astype(np.float64)


def cc_boxes(gray_word):
    """Connected-component letter boxes inside a word (separated Hebrew letters)."""
    inv = 255 - gray_word
    _, b = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    nlab, _, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    H = gray_word.shape[0]
    boxes = []
    for i in range(1, nlab):
        x, y, w, h, area = stats[i]
        if area > 0.001 * gray_word.size and h > 0.2 * H:    # drop specks / stray dots
            boxes.append((x, y, x + w, y + h))
    return boxes


def valley_cut(gray_word, x0, x1, margin=2):
    """Column of minimum ink inside [x0, x1) — where to split a too-wide box."""
    if x1 - x0 < 2 * margin + 1:
        return None
    _, prof = ink_profile(gray_word)
    seg = prof[x0:x1]
    interior = seg[margin:len(seg) - margin]
    if len(interior) == 0:
        return None
    return x0 + margin + int(np.argmin(interior))


def reconcile(boxes, gray_word, n):
    """Force a box list to exactly n: merge closest neighbours / split widest box."""
    H, W = gray_word.shape
    boxes = [list(b) for b in sorted(boxes, key=lambda b: b[0])]
    if not boxes:
        boxes = [[0, 0, W, H]]
    while len(boxes) > n:                       # too many -> merge the closest pair
        gaps = [boxes[i + 1][0] - boxes[i][2] for i in range(len(boxes) - 1)]
        i = int(np.argmin(gaps))
        a, b = boxes[i], boxes[i + 1]
        boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
        del boxes[i + 1]
    while len(boxes) < n:                       # too few -> split the widest at its valley
        i = int(np.argmax([b[2] - b[0] for b in boxes]))
        b = boxes[i]
        cut = valley_cut(gray_word, b[0], b[2]) or (b[0] + b[2]) // 2
        cut = min(max(cut, b[0] + 1), b[2] - 1)
        boxes[i:i + 1] = [[b[0], b[1], cut, b[3]], [cut, b[1], b[2], b[3]]]
        boxes.sort(key=lambda x: x[0])
    return boxes


def segment_word(gray_word, n):
    """Detect exactly n letter boxes: connected components, reconciled to n.
    Returns (boxes in word coords, detected_cc_count)."""
    if n <= 0:
        return [], 0
    boxes = cc_boxes(gray_word)
    return reconcile(boxes, gray_word, n), len(boxes)


def tighten(gray_seg):
    if gray_seg.size == 0:
        return gray_seg
    inv = 255 - gray_seg
    _, b = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.where(b > 0)
    if len(xs) == 0:
        return gray_seg
    return gray_seg[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def square_fit(img, size=None, pad_ratio=0.18, bg=255):
    size = size or IMG
    h, w = img.shape
    if h == 0 or w == 0:
        return np.full((size, size), bg, np.uint8)
    pad = int(max(h, w) * pad_ratio)
    side = max(h, w) + 2 * pad
    c = np.full((side, side), bg, np.uint8)
    c[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = img
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)


def candidate_divisions(gray_word, n):
    """Several ways to split the word into n letter boxes (each reconciled to exactly n).
    The iterative segmenter tries them and lets the classifier pick the best."""
    inv = 255 - gray_word
    _, m = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    H = gray_word.shape[0]
    k = np.ones((2, 2), np.uint8)

    def cc(mask):
        nlab, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        bx = [(x, y, x + w, y + h) for i, (x, y, w, h, a) in enumerate(stats)
              if i > 0 and a > 0.001 * gray_word.size and h > 0.2 * H]
        return reconcile(bx, gray_word, n)

    return [
        cc(m),                       # plain connected components
        cc(cv2.dilate(m, k)),        # merge over-split multi-stroke letters
        cc(cv2.erode(m, k)),         # separate lightly-touching letters
        reconcile([], gray_word, n), # pure projection-valley split
    ]


def segment_word_iter(gray_word, ocr_letters):
    """Iterative: try candidate divisions, grade each letter with the classifier
    (argmax == OCR letter AND prob >= LETTER_OK), keep the division with the most
    accepted letters. Returns (boxes RTL, [(letter, conf, ok)])."""
    n = len(ocr_letters)
    if n <= 0:
        return [], []
    best = None
    for boxes in candidate_divisions(gray_word, n):
        boxes_rtl = sorted(boxes, key=lambda b: -b[0])     # rightmost = first letter
        tiles = [square_fit(tighten(gray_word[max(0, y0):y1, max(0, x0):x1]))
                 for (x0, y0, x1, y1) in boxes_rtl]
        probs = classify(tiles)
        results = []
        for lab, p in zip(ocr_letters, probs):
            pl = float(p[CLS[lab]])        # classifier's probability for the OCR-assigned letter
            results.append((lab, pl, pl >= LETTER_OK))
        passed = sum(r[2] for r in results)
        score = (passed, sum(r[1] for r in results))   # most accepted, then total confidence
        if best is None or score > best[0]:
            best = (score, boxes_rtl, results)
        if passed == n:                # all letters clean -> stop trying divisions
            break
    return best[1], best[2]


# CRAFT char-detection attempts, ordered from moderate to aggressive splitting.
# (mag_ratio, text_threshold, link_threshold, low_text): higher mag/text/link -> more boxes.
# (mag_ratio, text_threshold, link_threshold, low_text). Region map is peak-normalized, so
# these are fractions of the model's peak. low_text is the SEPARATION knob — high low_text cuts
# adjacent letters' score-skirts apart into separate blobs; link kept high to avoid re-merging.
CHAR_CONFIGS = [
    (6, 0.40, 0.95, 0.45),
    (6, 0.40, 0.97, 0.55),
    (8, 0.40, 0.97, 0.60),
    (8, 0.45, 0.98, 0.65),
    (10, 0.45, 0.98, 0.70),
    (6, 0.50, 0.95, 0.50),
    (8, 0.35, 0.97, 0.55),
]


def craft_split_to_n(gray_word, n):
    """Try CRAFT with different thresholds until it returns exactly n char boxes.
    Returns (boxes RTL-sorted, attempts) on success, or (None, attempts) if none hit n."""
    word_rgb = cv2.cvtColor(gray_word, cv2.COLOR_GRAY2RGB)
    attempts = []
    for mag, text, link, low in CHAR_CONFIGS:
        st, sl, ratio = score_maps(NET_LETTERS, word_rgb, mag_ratio=mag)   # fine-tuned for letters
        peak = float(st.max())          # normalize region to its own peak (model output scale varies)
        if peak > 1e-6:
            st = st / peak
        boxes = [bbox(b) for b in boxes_from_scores(st, sl, ratio, text, link, low)]
        attempts.append(len(boxes))
        if len(boxes) == n:
            return sorted(boxes, key=lambda b: -b[0]), attempts   # rightmost = first letter
    return None, attempts


# ── core ─────────────────────────────────────────────────────────────────────
def analyze(image_bgr, use_ocr=True):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    st, sl, ratio = score_maps(NET, rgb)
    word_boxes = [bbox(b) for b in boxes_from_scores(st, sl, ratio, 0.7, 0.4, 0.4)]

    overlay = image_bgr.copy()
    for wb in word_boxes:
        cv2.rectangle(overlay, (wb[0], wb[1]), (wb[2], wb[3]), (0, 140, 255), 3)  # word: orange

    words = []
    best = {}            # letter -> (classifier_conf, crop_uri) : keep the most confident crop
    for wb in reading_order(word_boxes):
        x0, y0, x1, y1 = wb
        word_crop = gray[y0:y1, x0:x1]
        res = ocr_word(word_crop) if use_ocr else None
        text, conf = res if res else (None, None)
        used = bool(text) and conf is not None and conf >= CONF_MIN
        n = len([c for c in (text or "") if c in ALPHASET])

        seg_letters, matched, cc_n = [], False, 0
        if used and n > 0:
            boxes, cc_n = segment_word(word_crop, n)        # connected components, reconciled to n
            matched = (cc_n == n)                            # exact CC match vs reconciled
            boxes = sorted(boxes, key=lambda b: -b[0])       # right-to-left (first letter = rightmost)
            subs = [tighten(word_crop[max(0, b):d, max(0, a):c]) for (a, b, c, d) in boxes]
            tiles = [square_fit(s) for s in subs]
            probs = classify(tiles)                          # classifier only labels + scores confidence
            for (a, b, c, d), s, p in zip(boxes, subs, probs):
                j = int(p.argmax()); lconf = float(p[j]); lab = CLASSES[j]
                uri = png_b64(s)
                seg_letters.append((lab, lconf, uri))
                col = (0, 170, 0) if matched else (0, 160, 255)   # green=clean CC, orange=reconciled
                cv2.rectangle(overlay, (x0 + a, y0 + b), (x0 + c, y0 + d), col, 2)
                if lconf > best.get(lab, (0.0, None))[0]:
                    best[lab] = (lconf, uri)

        words.append({
            "crop": png_b64(word_crop),
            "ocr": text, "conf": conf, "n": n,
            "used": used, "matched": matched, "cc": cc_n, "letters": seg_letters,
        })
    return png_b64(overlay), words, best, len(words)


# ── HTML ─────────────────────────────────────────────────────────────────────
PAGE = """<!doctype html><html lang="he"><head><meta charset="utf-8">
<title>Hebrew Handwriting → TrOCR → Letters</title>
<style>
 body{{font-family:system-ui,Arial,sans-serif;margin:24px;background:#faf8f3;color:#222}}
 h1{{font-size:20px}} h2{{font-size:16px;margin-top:26px}} .muted{{color:#777}}
 form{{margin:16px 0;padding:16px;border:1px dashed #bbb;border-radius:10px;background:#fff}}
 .overlay img{{max-width:100%;border:1px solid #ccc;border-radius:8px}}
 .words{{display:flex;flex-wrap:wrap;gap:14px;margin-top:12px;direction:rtl}}
 .word{{border:1px solid #ddd;border-radius:10px;background:#fff;padding:10px}}
 .word.skip{{opacity:.5}}
 .word.fail{{border-color:#e88;background:#fff5f5}}
 .word .wimg{{height:54px;object-fit:contain;display:block;margin:0 auto 6px}}
 .reading{{font-size:24px;font-weight:700;text-align:center;margin-bottom:4px}}
 .chars{{display:flex;gap:6px;direction:rtl}}
 .chip{{text-align:center;width:52px}}
 .chip img{{width:44px;height:44px;object-fit:contain;border:1px solid #eee;border-radius:6px;background:#fff}}
 .chip .p{{font-size:18px;font-weight:700;line-height:1.1}}
 .chip .c{{font-size:10px;color:#999}}
 .grid{{display:flex;flex-wrap:wrap;gap:10px;margin-top:12px;direction:rtl}}
 .cell{{width:92px;border:1px solid #ddd;border-radius:8px;background:#fff;padding:8px;text-align:center}}
 .letter{{font-size:28px;font-weight:700;line-height:1}} .found{{color:#2a7}}
 .cell img{{width:60px;height:60px;object-fit:contain;margin-top:6px}}
 .none{{width:60px;height:60px;display:flex;align-items:center;justify-content:center;color:#ccc;margin:6px auto 0;border:1px dashed #eee;border-radius:6px}}
 .conf{{font-size:11px;color:#888;margin-top:4px}}
</style></head><body>
<h1>Hebrew Handwriting → TrOCR → Letters <span class="muted">(words ≥ {conf_pct}% OCR confidence; letters by CRAFT, scored by classifier)</span></h1>
<form action="/analyze" method="post" enctype="multipart/form-data">
  <input type="file" name="file" accept="image/*" required>
  <label style="margin-left:12px">Preprocess:
    <select name="prep">
      <option value="none">none</option>
      <option value="clean" selected>clean</option>
      <option value="binarize">binarize (Sauvola)</option>
    </select></label>
  <label style="margin-left:8px"><input type="checkbox" name="deskew" value="on" checked> deskew</label>
  <button type="submit" style="margin-left:8px">Analyze</button>
</form>
{result}
</body></html>"""


def render(result=""):
    return PAGE.format(result=result, conf_pct=int(CONF_MIN * 100))


@app.get("/", response_class=HTMLResponse)
def index():
    return render()


@app.post("/analyze", response_class=HTMLResponse)
async def do_analyze(file: UploadFile = File(...),
                     prep: str = Form("clean"),
                     deskew: str = Form("off")):
    data = await file.read()
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return render('<p style="color:#c00">Could not read that image.</p>')
    prepped = preprocess(arr, mode=prep, do_deskew=(deskew == "on"))
    prep_uri = png_b64(prepped)
    overlay_uri, words, best, nw = analyze(cv2.cvtColor(prepped, cv2.COLOR_GRAY2BGR))

    ocr_on = any(w["ocr"] is not None for w in words)
    banner = ('<p style="color:#2a7">TrOCR connected (<b>trocr-hebrew-matan-exp7</b>).</p>' if ocr_on
              else '<p style="color:#c00">TrOCR server not reachable on :8001 — start it; this mode needs it.</p>')
    used_n = sum(w["used"] for w in words)
    matched_n = sum(w["matched"] for w in words)

    word_html = []
    for w in words:
        if w["conf"] is not None:
            badge = (f'<div style="text-align:center;font-size:12px;font-weight:700;'
                     f'color:{conf_color(w["conf"])}">{w["conf"]*100:.0f}%</div>')
        else:
            badge = ""
        if w["used"]:
            chips = "".join(
                f'<div class="chip"><div class="p">{ch}</div><img src="{uri}">'
                f'<div class="c" style="color:{conf_color(lconf)}">{lconf*100:.0f}%</div></div>'
                for ch, lconf, uri in w["letters"])
            note = (f'<div class="muted" style="text-align:center;font-size:10px">'
                    f'CC found {w["cc"]} → {"matched" if w["matched"] else "reconciled to"} {w["n"]}</div>')
            body = note + f'<div class="chars">{chips}</div>'
            cls = "word"
        else:
            reason = "conf &lt; %d%%" % int(CONF_MIN * 100) if w["conf"] is not None else "no OCR"
            body = f'<div class="muted" style="text-align:center;font-size:11px">skipped ({reason})</div>'
            cls = "word skip"
        word_html.append(
            f'<div class="{cls}"><img class="wimg" src="{w["crop"]}">'
            f'<div class="reading">{w["ocr"] or "—"}</div>{badge}{body}</div>')

    cells = []
    for letter in ALPHABET:
        if letter in best:
            conf, uri = best[letter]
            cells.append(f'<div class="cell"><div class="letter found">{letter}</div>'
                         f'<img src="{uri}"><div class="conf">{conf*100:.0f}%</div></div>')
        else:
            cells.append(f'<div class="cell"><div class="letter">{letter}</div>'
                         f'<div class="none">—</div><div class="conf">not found</div></div>')

    result = (f'<p><b>{nw}</b> words · <b>{used_n}</b> ≥{int(CONF_MIN*100)}% OCR · '
              f'<b>{matched_n}</b> CC matched N · <b>{len(best)}/27</b> letters · '
              f'preprocess=<b>{prep}</b></p>{banner}'
              f'<h2>Preprocessed</h2><div class="overlay"><img src="{prep_uri}"></div>'
              f'<h2>Detection &amp; splits</h2><div class="overlay"><img src="{overlay_uri}"></div>'
              f'<h2>Words</h2><div class="words">{"".join(word_html)}</div>'
              f'<h2>Alphabet</h2><div class="grid">{"".join(cells)}</div>')
    return render(result)
