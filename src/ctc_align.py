"""CTC forced alignment — split a word image into per-character bounding boxes.

Given a grayscale word crop and its known Hebrew text (from TrOCR), this module:
  1. Runs the trained CRNN to get per-timestep log-probabilities.
  2. Finds the most-probable CTC path that produces exactly that text (Viterbi).
  3. Maps each character's time-step range back to pixel x-coordinates.

Public API (drop-in for segment_word in app.py):

    boxes, n_labels = split_word(gray_word, text, model=None, device="cpu")

    boxes   — list of (x0, y0, x1, y1) sorted left-to-right in image coords.
              (index 0 = leftmost in image = LAST letter in Hebrew reading order)
    n_labels — number of Hebrew characters in `text` (mirrors the cc_count
               return value so callers don't need changes)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from crnn import CRNN, NUM_CLASSES, encode, BLANK

# ── model loading ─────────────────────────────────────────────────────────────

_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "crnn.pth"
_cached_model: Optional[CRNN] = None


def load_crnn(ckpt: str | Path = _MODEL_PATH, device: str = "cpu") -> CRNN:
    ckpt = Path(ckpt)
    state = torch.load(ckpt, map_location=device)
    model = CRNN(
        num_classes=state.get("num_classes", NUM_CLASSES),
        hidden=state.get("hidden", 256),
    )
    model.load_state_dict(state["model_state"])
    model.eval()
    return model.to(device)


def _get_model(model: Optional[CRNN], device: str) -> CRNN:
    global _cached_model
    if model is not None:
        return model
    if _cached_model is None:
        _cached_model = load_crnn(device=device)
    return _cached_model


# ── image preprocessing ───────────────────────────────────────────────────────

_IMG_H = 64
_MIN_W = 32


def _prepare(gray: np.ndarray) -> tuple[torch.Tensor, int]:
    """Resize to H=64, return tensor (1,1,H,W) and original-pixel width (pre-pad)."""
    oh, ow = gray.shape[:2]
    new_w = max(_MIN_W, int(round(ow * _IMG_H / oh)))
    resized = cv2.resize(gray, (new_w, _IMG_H), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(resized).float().div_(255).unsqueeze(0).unsqueeze(0)
    return t, new_w


# ── Viterbi CTC forced alignment ──────────────────────────────────────────────

def _viterbi_align(log_probs: np.ndarray, labels: list[int]) -> list[float]:
    """Find the most-probable CTC path that decodes to `labels`.

    log_probs : (T, C) array of log-probabilities (blank = index 0)
    labels    : list of target character indices (no blanks)

    Returns the per-character spike CENTER (timestep, float), one per label.
    """
    T, _ = log_probs.shape
    N = len(labels)
    if N == 0:
        return []

    # CTC label sequence: blank l1 blank l2 … lN blank
    ctc = [BLANK]
    for l in labels:
        ctc.append(l)
        ctc.append(BLANK)
    S = len(ctc)  # = 2N + 1

    NEG_INF = -1e30
    alpha = np.full((T, S), NEG_INF, dtype=np.float64)
    back  = np.zeros((T, S), dtype=np.int32)

    # t = 0: only positions 0 and 1 reachable
    alpha[0, 0] = log_probs[0, BLANK]
    if N > 0:
        alpha[0, 1] = log_probs[0, labels[0]]

    for t in range(1, T):
        for s in range(S):
            c = ctc[s]

            # Candidate predecessor positions
            candidates = [s]                     # stay
            if s > 0:
                candidates.append(s - 1)         # advance one step
            if (s > 1
                    and ctc[s] != BLANK          # skip blank only for non-blank
                    and ctc[s] != ctc[s - 2]):   # and only if char differs
                candidates.append(s - 2)

            best_s = max(candidates, key=lambda p: alpha[t - 1, p])
            alpha[t, s] = alpha[t - 1, best_s] + log_probs[t, c]
            back[t, s]  = best_s

    # Pick best end state (last blank or last label)
    end = S - 1 if alpha[T - 1, S - 1] >= alpha[T - 1, S - 2] else S - 2

    # Backtrack
    path = np.empty(T, dtype=np.int32)
    path[T - 1] = end
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]

    # CTC is "peaky": each character is emitted at just 1-2 spike timesteps,
    # with blanks in between. Using only the spike gives narrow slices. Instead
    # take each character's spike CENTER, then place box boundaries at the
    # midpoints between consecutive centers so the boxes tile the full width.
    centers: list[float] = []
    for i in range(N):
        char_pos = 2 * i + 1
        times = np.where(path == char_pos)[0]
        if len(times) > 0:
            centers.append(float(times.mean()))
        else:
            # Character collapsed (no timestep) — interpolate its fair-share center
            centers.append((i + 0.5) * T / N)

    # Guard against non-monotonic / collapsed centers → fall back to equal split
    if any(centers[i] >= centers[i + 1] for i in range(N - 1)):
        centers = [(i + 0.5) * T / N for i in range(N)]

    return centers


# ── main public function ──────────────────────────────────────────────────────

def split_word(
    gray_word: np.ndarray,
    text: str,
    model: Optional[CRNN] = None,
    device: str = "cpu",
    pad_frac: float = 0.25,
) -> tuple[list[tuple[int, int, int, int]], int]:
    """Segment a grayscale word image into per-character bounding boxes.

    Parameters
    ----------
    gray_word : (H, W) uint8 grayscale crop
    text      : Hebrew word string in natural reading order (RTL)
    model     : optional pre-loaded CRNN (avoids repeated disk loads)
    device    : 'cpu' or 'cuda'
    pad_frac  : how much to widen each box past the midpoint cut, as a fraction
                of the box width, on EACH side. Boxes are allowed to overlap so
                slanted/skewed letters keep strokes that lean into a neighbour.
                0.0 = hard non-overlapping midpoint cuts; 0.25 = +25% each side.

    Returns
    -------
    boxes : list of (x0, y0, x1, y1) in word-image pixel coords,
            sorted LEFT-TO-RIGHT (= right-to-left in Hebrew reading order).
            Adjacent boxes may overlap when pad_frac > 0.
    n     : number of Hebrew characters found in `text`
    """
    # Encode: reverse text so CTC sees characters in left-to-right scan order
    labels = encode(text[::-1])
    n = len(labels)
    if n == 0:
        return [], 0

    H, W = gray_word.shape[:2]

    # Run CRNN
    net = _get_model(model, device)
    tensor, feat_w = _prepare(gray_word)
    tensor = tensor.to(device)
    with torch.no_grad():
        log_probs = net(tensor)   # (T, 1, C)
    log_probs_np = log_probs[:, 0, :].cpu().numpy()  # (T, C)
    T = log_probs_np.shape[0]

    # Viterbi forced alignment → per-character spike centers (timestep units)
    centers = _viterbi_align(log_probs_np, labels)

    # Map time-step centers → pixel x-coordinates in the ORIGINAL word image
    scale = W / T
    cx = [c * scale for c in centers]

    # Each box is sized from the LOCAL spacing around its spike (half the gap to
    # each neighbour) and centered on the spike. The outer edges are NOT anchored
    # to the word rectangle, so boxes do not fill/stretch to the word width — the
    # first/last letters keep a width mirrored from their inner gap.
    boxes: list[tuple[int, int, int, int]] = []
    for i in range(n):
        if n == 1:
            left_gap = right_gap = float(W)
        else:
            left_gap  = (cx[i] - cx[i - 1]) if i > 0     else (cx[i + 1] - cx[i])
            right_gap = (cx[i + 1] - cx[i]) if i < n - 1 else (cx[i] - cx[i - 1])
        x0f = cx[i] - left_gap / 2.0
        x1f = cx[i] + right_gap / 2.0
        # Optional widening so skewed letters aren't clipped (boxes may overlap).
        pad = (x1f - x0f) * pad_frac
        x0 = max(0, int(round(x0f - pad)))
        x1 = min(W, int(round(x1f + pad)))
        if x1 <= x0:
            x1 = min(W, x0 + 1)
        boxes.append((x0, 0, x1, H))

    # boxes are in LTR image order (leftmost = last letter in reading order)
    return sorted(boxes, key=lambda b: b[0]), n
