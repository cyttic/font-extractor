#!/usr/bin/env python3
"""CRAFT word-detection server — runs LOCALLY (notebook), where CRAFT is fast.

The font-extractor frontend on the Azure VM is thin: it sends a page image here
through the reverse SSH tunnel (VM:9001 -> notebook:9001), gets back word bounding
boxes as JSON, then draws the rectangles and runs OCR. This keeps torch and the
CRAFT weights OFF the VM (same pattern as the TrOCR model server on :8001).

Run from the repo root so models/ and third_party/ resolve:
    uvicorn src.craft_server:app --host 127.0.0.1 --port 9001
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from run_craft import load_net, score_maps, boxes_from_scores

app = FastAPI(title="CRAFT word-detection server")

print("loading CRAFT ...")
NET = load_net()                 # stock craft_mlt_25k.pth (clovaai general model)
print("ready.")


def _bbox(poly):
    p = np.array(poly)
    return [int(p[:, 0].min()), int(p[:, 1].min()),
            int(p[:, 0].max()), int(p[:, 1].max())]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    text_threshold: float = Form(0.7),
    link_threshold: float = Form(0.4),
    low_text: float = Form(0.4),
):
    """Return {"boxes": [[x0,y0,x1,y1], ...]} — axis-aligned word boxes in the
    uploaded image's pixel coordinates (same space the frontend draws on)."""
    data = await file.read()
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)   # BGR
    if arr is None:
        raise HTTPException(400, "could not decode image")
    # match the old in-frontend pipeline: gray -> RGB -> CRAFT
    rgb = cv2.cvtColor(cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
    st, sl, ratio = score_maps(NET, rgb)
    polys = boxes_from_scores(st, sl, ratio, text_threshold, link_threshold, low_text)
    return {"boxes": [_bbox(p) for p in polys]}
