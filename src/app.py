#!/usr/bin/env python3
"""FastAPI demo — TrOCR word finder.

Pipeline:
  preprocess -> CRAFT splits the page into WORDS (reading order, RTL + top-to-bottom)
  -> TrOCR reads each word + confidence.
Each word box is colored by OCR confidence:
  >= 90% green · < 30% red · otherwise orange.

    .venv/bin/uvicorn src.app:app --reload --port 8000   ->  http://localhost:8000
    (needs the TrOCR server on :8001 — see webHebrewOCR)
"""
import base64
import io
import sys
from pathlib import Path

import cv2
import numpy as np
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from preprocess import preprocess

OCR_URL = "http://127.0.0.1:8001"
OCR_MODEL = "trocr-hebrew-matan-exp7"
OCR_BEAMS = 4

# CRAFT word detection runs OFF the VM, served by the SAME backend as OCR on
# :8001 (one reverse SSH tunnel). The model server exposes a /detect endpoint.
CRAFT_URL = "http://127.0.0.1:8001"

# confidence thresholds for the word border
CONF_GREEN = 0.90
CONF_RED = 0.30

app = FastAPI(title="Hebrew Handwriting → TrOCR → Words")

# decoded BGR of the most recent upload, so "Analyze" can re-run without re-picking a file
LAST_IMAGE = None


# ── OCR client ───────────────────────────────────────────────────────────────
def ocr_available() -> bool:
    try:
        return requests.get(f"{OCR_URL}/health", timeout=2).status_code == 200
    except Exception:
        return False


def ocr_models() -> list[str]:
    """List the recognition models the OCR server exposes (falls back to default)."""
    try:
        r = requests.get(f"{OCR_URL}/models", timeout=2)
        if r.status_code == 200:
            ms = r.json().get("models") or []
            if ms:
                return ms
    except Exception:
        pass
    return [OCR_MODEL]


def ocr_word(gray_crop, model: str = OCR_MODEL):
    ok, buf = cv2.imencode(".png", gray_crop)
    try:
        r = requests.post(f"{OCR_URL}/ocr",
                          data={"model": model, "beams": OCR_BEAMS},
                          files={"file": ("word.png", buf.tobytes(), "image/png")},
                          timeout=30)
        if r.status_code != 200:
            return None
        j = r.json()
        return j.get("text", ""), float(j.get("confidence", 0.0))
    except Exception:
        return None


# ── CRAFT client (remote word detection) ─────────────────────────────────────
def craft_available() -> bool:
    try:
        return requests.get(f"{CRAFT_URL}/health", timeout=2).status_code == 200
    except Exception:
        return False


def detect_words(image_bgr) -> list[tuple[int, int, int, int]]:
    """POST the page to the remote CRAFT server; get back word bounding boxes
    (x0, y0, x1, y1) in this image's pixel coordinates."""
    ok, buf = cv2.imencode(".png", image_bgr)
    try:
        r = requests.post(f"{CRAFT_URL}/detect",
                          files={"file": ("page.png", buf.tobytes(), "image/png")},
                          timeout=60)
        if r.status_code != 200:
            return []
        return [tuple(int(v) for v in b) for b in r.json().get("boxes", [])]
    except Exception:
        return []


def word_colors(c):
    """(css_hex, bgr_tuple) for a confidence value: green / orange / red / gray."""
    if c is None:
        return "#bbbbbb", (187, 187, 187)        # no OCR
    if c >= CONF_GREEN:
        return "#22aa77", (119, 170, 34)         # green
    if c < CONF_RED:
        return "#cc0000", (0, 0, 204)            # red
    return "#e0a000", (0, 160, 224)              # orange


# ── geometry helpers ─────────────────────────────────────────────────────────
def cx(b): return (b[0] + b[2]) / 2
def cy(b): return (b[1] + b[3]) / 2


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


# ── core ─────────────────────────────────────────────────────────────────────
def analyze(image_bgr, model: str = OCR_MODEL):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    word_boxes = detect_words(image_bgr)               # remote CRAFT -> word boxes

    overlay = image_bgr.copy()
    words = []
    for wb in reading_order(word_boxes):
        x0, y0, x1, y1 = wb
        word_crop = gray[y0:y1, x0:x1]
        res = ocr_word(word_crop, model=model)
        text, conf = res if res else (None, None)
        _, col_bgr = word_colors(conf)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), col_bgr, 3)
        words.append({"crop": png_b64(word_crop), "ocr": text, "conf": conf})
    return png_b64(overlay), words, len(words)


# ── HTML ─────────────────────────────────────────────────────────────────────
PAGE = """<!doctype html><html lang="he"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hebrew Handwriting → TrOCR → Words</title>
<style>
 body{{font-family:system-ui,Arial,sans-serif;margin:24px;background:#faf8f3;color:#222}}
 h1{{font-size:20px}} h2{{font-size:16px;margin-top:26px}} .muted{{color:#777}}
 form{{margin:16px 0;padding:16px;border:1px dashed #bbb;border-radius:10px;background:#fff}}
 .overlay img{{max-width:100%;border:1px solid #ccc;border-radius:8px}}
 .words{{display:flex;flex-wrap:wrap;gap:14px;margin-top:12px;direction:rtl}}
 .word{{border:3px solid #ddd;border-radius:10px;background:#fff;padding:10px;text-align:center}}
 .word .wimg{{height:54px;object-fit:contain;display:block;margin:0 auto 6px}}
 .reading{{font-size:24px;font-weight:700;margin-bottom:4px}}
 .badge{{font-size:13px;font-weight:700}}
 .legend{{font-size:12px}} .legend b{{padding:1px 6px;border-radius:4px;color:#fff}}
 .drop{{border:2px dashed #bbb;border-radius:10px;padding:18px;text-align:center;cursor:pointer;background:#fafafa;transition:.15s;margin-bottom:10px}}
 .drop.drag{{border-color:#2a7;background:#eefaf2}}
 .drop .hint{{color:#777;font-size:13px}} .drop .fname{{margin-top:6px;font-weight:600;color:#2a7}}
 #ov{{position:fixed;inset:0;background:rgba(255,255,255,.86);display:none;align-items:center;justify-content:center;flex-direction:column;z-index:99}}
 #ov.on{{display:flex}}
 .spin{{width:48px;height:48px;border:5px solid #ddd;border-top-color:#2a7;border-radius:50%;animation:sp 1s linear infinite}}
 @keyframes sp{{to{{transform:rotate(360deg)}}}}
 #ovmsg{{margin-top:14px;font-weight:700}}
 button{{background:#2a7;color:#fff;border:0;cursor:pointer;border-radius:8px;padding:8px 14px;font-size:15px}}
 select{{font-size:15px;padding:6px 8px;border-radius:8px}}
 .cam{{margin-top:8px;background:#fff;color:#2a7;border:1px solid #2a7;border-radius:8px;padding:8px 14px;font-size:15px;cursor:pointer}}
 .controls{{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-top:12px}}
 .controls label{{display:inline-flex;align-items:center;gap:6px}}
 @media (max-width:600px){{
   body{{margin:12px}} h1{{font-size:18px}}
   .controls{{flex-direction:column;align-items:stretch}}
   .controls label,.controls select,.controls button,.cam{{width:100%}}
   .word{{flex:1 1 100%}} .word .wimg{{height:64px}}
 }}
</style></head><body>
<div id="ov"><div class="spin"></div><div id="ovmsg">Working…</div></div>
<h1>Hebrew Handwriting → TrOCR → Words
 <span class="muted">(CRAFT finds words; TrOCR reads them; border = OCR confidence)</span></h1>
<p class="legend">
 <b style="background:#22aa77">≥ 90% green</b>
 <b style="background:#e0a000">30–90% orange</b>
 <b style="background:#cc0000">&lt; 30% red</b></p>
<form id="f" action="/analyze" method="post" enctype="multipart/form-data">
  <div class="drop" id="drop">
    <div class="hint">Drag &amp; drop an image here, or click to choose a file</div>
    <div class="fname" id="fname"></div>
    <input type="file" id="file" name="file" accept="image/*" style="display:none">
  </div>
  <button type="button" class="cam" id="camBtn">📷 Take photo</button>
  <input type="file" id="cam" accept="image/*" capture="environment" style="display:none">
  <div class="muted" style="font-size:11px;margin-bottom:6px">(leave empty to re-run the last image)</div>
  <div class="controls">
    <label>OCR model:
      <select name="model">{model_options}</select></label>
    <label>Preprocess:
      <select name="prep">
        <option value="clean">clean</option>
        <option value="binarize" selected>binarize (Sauvola)</option>
      </select></label>
    <label><input type="checkbox" name="deskew" value="on" checked> deskew</label>
    <button type="submit">Analyze</button>
  </div>
</form>
{result}
<script>
(function(){{
  var drop=document.getElementById('drop'),file=document.getElementById('file'),
      fname=document.getElementById('fname'),form=document.getElementById('f'),
      ov=document.getElementById('ov'),msg=document.getElementById('ovmsg');
  function show(){{ fname.textContent = file.files.length ? ('✓ '+file.files[0].name) : ''; }}
  drop.addEventListener('click', function(){{ file.click(); }});
  file.addEventListener('change', show);
  ['dragenter','dragover'].forEach(function(e){{ drop.addEventListener(e, function(ev){{ ev.preventDefault(); drop.classList.add('drag'); }}); }});
  ['dragleave','drop'].forEach(function(e){{ drop.addEventListener(e, function(ev){{ ev.preventDefault(); drop.classList.remove('drag'); }}); }});
  drop.addEventListener('drop', function(ev){{ if(ev.dataTransfer.files.length){{ file.files = ev.dataTransfer.files; show(); }} }});
  var cam=document.getElementById('cam'),camBtn=document.getElementById('camBtn');
  camBtn.addEventListener('click', function(){{ cam.click(); }});
  cam.addEventListener('change', function(){{ if(cam.files.length){{ file.files = cam.files; show(); }} }});
  var stages=['Preprocessing image…','Detecting words (CRAFT)…','Recognizing text (OCR)…','Almost done…'],i=0;
  form.addEventListener('submit', function(){{ i=0; msg.textContent=stages[0]; ov.classList.add('on'); setInterval(function(){{ i=Math.min(i+1,stages.length-1); msg.textContent=stages[i]; }},1500); }});
}})();
</script>
</body></html>"""


def render(result="", model: str = OCR_MODEL):
    opts = "".join(
        f'<option value="{m}"{" selected" if m == model else ""}>{m}</option>'
        for m in ocr_models())
    return PAGE.format(result=result, model_options=opts)


@app.get("/", response_class=HTMLResponse)
def index():
    return render()


# Phone photos are huge (often 12 MP); the small VM OOMs building the overlay +
# base64. Cap the longest side so detection / preview / encoding stay light.
_MAX_SIDE = 1800


def _fit(arr):
    if arr is None:
        return None
    h, w = arr.shape[:2]
    m = max(h, w)
    if m > _MAX_SIDE:
        s = _MAX_SIDE / float(m)
        arr = cv2.resize(arr, (max(1, round(w * s)), max(1, round(h * s))),
                         interpolation=cv2.INTER_AREA)
    return arr


def _decode(data: bytes):
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        # OpenCV can't decode some TIFFs (e.g. old-style JPEG compression);
        # fall back to Pillow, which handles them.
        try:
            from PIL import Image
            pil = Image.open(io.BytesIO(data)).convert("RGB")
            arr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            arr = None
    return _fit(arr)


@app.post("/analyze", response_class=HTMLResponse)
async def do_analyze(file: UploadFile = File(None),
                     model: str = Form(OCR_MODEL),
                     prep: str = Form("binarize"),
                     deskew: str = Form("off")):
    global LAST_IMAGE
    arr = None
    if file is not None:
        data = await file.read()
        if data:
            arr = _decode(data)
            if arr is None:
                return render('<p style="color:#c00">Could not read that image.</p>', model=model)
            LAST_IMAGE = arr
    if arr is None:                          # no new file -> re-run the previous image
        arr = LAST_IMAGE
    if arr is None:
        return render('<p style="color:#c00">Upload an image to analyze.</p>', model=model)

    try:
        prepped = preprocess(arr, mode=prep, do_deskew=(deskew == "on"))
        prep_uri = png_b64(prepped)
        overlay_uri, words, nw = analyze(cv2.cvtColor(prepped, cv2.COLOR_GRAY2BGR), model=model)
    except Exception as e:
        return render(f'<p style="color:#c00">Processing failed: {type(e).__name__}: {e}</p>', model=model)

    craft_on = craft_available()
    ocr_on = ocr_available()
    craft_banner = ('<p style="color:#2a7">CRAFT connected (:8001 /detect).</p>' if craft_on else
                    '<p style="color:#c00">CRAFT not reachable on :8001 /detect — '
                    'restart the backend model server (it now also runs CRAFT).</p>')
    ocr_banner = (f'<p style="color:#2a7">TrOCR connected (<b>{model}</b>).</p>' if ocr_on
                  else '<p style="color:#c00">TrOCR server not reachable on :8001 — start it; this mode needs it.</p>')
    banner = craft_banner + ocr_banner
    green_n = sum(1 for w in words if w["conf"] is not None and w["conf"] >= CONF_GREEN)

    word_html = []
    for w in words:
        hexc, _ = word_colors(w["conf"])
        badge = (f'{w["conf"]*100:.0f}%' if w["conf"] is not None else "no OCR")
        word_html.append(
            f'<div class="word" style="border-color:{hexc}">'
            f'<img class="wimg" src="{w["crop"]}">'
            f'<div class="reading">{w["ocr"] or "—"}</div>'
            f'<div class="badge" style="color:{hexc}">{badge}</div></div>')

    result = (f'<p><b>{nw}</b> words · <b>{green_n}</b> ≥{int(CONF_GREEN*100)}% · '
              f'model=<b>{model}</b> · preprocess=<b>{prep}</b></p>{banner}'
              f'<h2>Preprocessed</h2><div class="overlay"><img src="{prep_uri}"></div>'
              f'<h2>Detection</h2><div class="overlay"><img src="{overlay_uri}"></div>'
              f'<h2>Words</h2><div class="words">{"".join(word_html)}</div>')
    return render(result, model=model)
