#!/usr/bin/env python3
"""Image preprocessing before detection/recognition.

Selectable modes (output is always black-ink-on-white, the polarity everything expects):
  none     — grayscale only (no cleanup)
  clean    — deskew + illumination flattening + denoise (grayscale)   [default]
  binarize — clean + Sauvola adaptive threshold (pure black/white)

Sauvola (per the build spec) beats global Otsu on uneven handwriting scans. Deskew uses a
projection-profile angle search (sharpest horizontal text rows). Illumination flattening
divides out a blurred background, removing shadows/gradients from photos.
"""
import cv2
import numpy as np


def _to_gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def deskew(gray, max_deg=5.0, step=0.5):
    """Rotate to maximize sharpness of the horizontal projection profile (text rows)."""
    inv = 255 - gray
    _, b = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = b.shape
    best_ang, best_score = 0.0, -1.0
    for ang in np.arange(-max_deg, max_deg + 1e-6, step):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        rot = cv2.warpAffine(b, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        proj = rot.sum(axis=1, dtype=np.float64)
        score = float(((proj[1:] - proj[:-1]) ** 2).sum())   # sharper rows -> higher
        if score > best_score:
            best_score, best_ang = score, float(ang)
    if abs(best_ang) < 1e-3:
        return gray
    M = cv2.getRotationMatrix2D((w / 2, h / 2), best_ang, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def flatten_illumination(gray):
    """Divide out a blurred background to remove shadows / uneven lighting."""
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(gray.shape) / 30.0)
    norm = gray.astype(np.float32) / (bg.astype(np.float32) + 1e-6)
    norm = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    return norm


def sauvola(gray, window=25, k=0.2, R=128.0):
    """Sauvola adaptive threshold via OpenCV box filters (no scikit-image dep).

    T(x,y) = mean * (1 + k*(std/R - 1)).  dark text -> 0, background -> 255.
    """
    win = window if window % 2 == 1 else window + 1
    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, -1, (win, win), borderType=cv2.BORDER_REPLICATE)
    mean_sq = cv2.boxFilter(g * g, -1, (win, win), borderType=cv2.BORDER_REPLICATE)
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    th = mean * (1.0 + k * (std / R - 1.0))
    return (g > th).astype(np.uint8) * 255


def preprocess(img, mode="clean", do_deskew=True):
    """img: BGR or gray. Returns a single-channel black-on-white uint8 image."""
    gray = _to_gray(img)
    if do_deskew and mode != "none":
        gray = deskew(gray)
    if mode == "none":
        return gray
    gray = flatten_illumination(gray)
    gray = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
    if mode == "binarize":
        gray = sauvola(gray)
    return gray


if __name__ == "__main__":
    import sys
    from pathlib import Path
    out = Path(__file__).resolve().parent.parent / "output" / "overlays"
    out.mkdir(parents=True, exist_ok=True)
    for p in sys.argv[1:]:
        img = cv2.imread(p)
        for mode in ("none", "clean", "binarize"):
            res = preprocess(img, mode=mode)
            cv2.imwrite(str(out / f"prep_{Path(p).stem}_{mode}.png"), res)
            print(f"{Path(p).name} [{mode}] -> shape {res.shape}")
