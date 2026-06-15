#!/usr/bin/env python3
"""Phase 1 — synthetic data generator.

Renders the 40 Hebrew handwriting fonts into two labeled datasets:

  * isolated single-glyph images  -> classifier (P4)
  * multi-letter line images + per-char boxes + CRAFT region/affinity heatmaps -> CRAFT (P2)

Both are written as tar/WebDataset shards (Kaggle-friendly) plus manifests.
The 40 fonts are scaffolding only; a rendered glyph must never reach the final TTF.

Per the SynthText trick: we *place* every glyph, so we know its exact bounding box.

Usage:
    python src/synth_data.py                      # full run from config.yaml
    python src/synth_data.py --per-class 4 --lines 50 --fonts 3   # quick smoke run
"""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path

import cv2
import numpy as np
import webdataset as wds
import yaml
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent


# ─── config ──────────────────────────────────────────────────────────────────
def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(root: Path, p: str) -> Path:
    """Resolve a config path: absolute stays, relative is under repo root."""
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp)


# ─── glyph rendering ─────────────────────────────────────────────────────────
def render_glyph(font: ImageFont.FreeTypeFont, char: str) -> np.ndarray | None:
    """Render one character as a tight ink-on-black uint8 raster (255 = ink).

    Returns None if the font produces no ink for this codepoint.
    """
    # Generous canvas so nothing clips, then crop to the ink bbox.
    asc, desc = font.getmetrics()
    box = font.getbbox(char)
    if box is None:
        return None
    w = max(box[2] - box[0], 1) + 8
    h = asc + desc + 8
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    d.text((4 - box[0], 4), char, font=font, fill=255)
    arr = np.asarray(img)
    ys, xs = np.where(arr > 10)
    if len(xs) == 0:
        return None
    return arr[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def has_glyph(font_path: Path, char: str) -> bool:
    from fontTools.ttLib import TTFont
    try:
        cmap = TTFont(str(font_path)).getBestCmap()
        return ord(char) in cmap
    except Exception:
        return False


# ─── augmentation (seeded, toggleable) ───────────────────────────────────────
class Augmenter:
    """Operates on an ink map (float [0,1], 1 = ink). All ops are seeded."""

    def __init__(self, cfg: dict, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def _u(self, lo, hi):
        return float(self.rng.uniform(lo, hi))

    def elastic(self, img):
        c = self.cfg["elastic"]
        if not c["enabled"]:
            return img
        h, w = img.shape
        alpha = self._u(*c["alpha"])
        sigma = self._u(*c["sigma"])
        dx = cv2.GaussianBlur(self.rng.random((h, w)).astype(np.float32) * 2 - 1, (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur(self.rng.random((h, w)).astype(np.float32) * 2 - 1, (0, 0), sigma) * alpha
        gx, gy = np.meshgrid(np.arange(w), np.arange(h))
        mx = (gx + dx).astype(np.float32)
        my = (gy + dy).astype(np.float32)
        return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def affine(self, img):
        h, w = img.shape
        M = np.eye(2, 3, dtype=np.float32)
        if self.cfg["slant"]["enabled"]:
            M[0, 1] = self._u(-1, 1) * self.cfg["slant"]["max_shear"]
        if self.cfg["rotation"]["enabled"]:
            ang = self._u(-1, 1) * self.cfg["rotation"]["max_deg"]
            R = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
            img = cv2.warpAffine(img, R, (w, h), borderValue=0)
        return cv2.warpAffine(img, M, (w, h), borderValue=0)

    def stroke(self, img):
        c = self.cfg["stroke"]
        if not c["enabled"]:
            return img
        op = self.rng.choice(c["op"])
        if op == "none":
            return img
        k = int(self.rng.integers(1, c["max_kernel"] + 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        fn = cv2.erode if op == "erode" else cv2.dilate
        return fn(img, kernel)

    def blur(self, img):
        c = self.cfg["blur"]
        if not c["enabled"]:
            return img
        s = self._u(0.01, c["max_sigma"])
        return cv2.GaussianBlur(img, (0, 0), s)

    def glyph(self, ink: np.ndarray) -> np.ndarray:
        """Full per-glyph augmentation chain. Input/output ink map float [0,1]."""
        if not self.cfg["enabled"]:
            return ink
        ink = self.elastic(ink)
        ink = self.affine(ink)
        ink = self.stroke(ink)
        if self.cfg["ink_bleed"]["enabled"] and self.rng.random() < self.cfg["ink_bleed"]["prob"]:
            ink = cv2.dilate(ink, np.ones((2, 2), np.uint8).astype(np.float32))
            ink = cv2.GaussianBlur(ink, (0, 0), 0.6)
        ink = self.blur(ink)
        return np.clip(ink, 0, 1)


def compose_on_paper(ink: np.ndarray, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """Composite ink map -> black-ink-on-white uint8 (matches scanned crops)."""
    h, w = ink.shape
    if cfg["paper_bg"]["enabled"] and rng.random() < cfg["paper_bg"]["prob"]:
        bg = rng.uniform(0.85, 1.0)
        tex = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 3)
        paper = np.clip(bg - 0.06 * (tex - tex.mean()), 0, 1)
    else:
        paper = np.ones((h, w), np.float32)
    dark = 0.05
    out = paper * (1 - ink) + dark * ink
    if cfg["noise"]["enabled"]:
        std = rng.uniform(*cfg["noise"]["gauss_std"]) / 255.0
        out = out + rng.normal(0, std, out.shape)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def square_fit(img: np.ndarray, size: int, pad_ratio: float, bg: int = 255) -> np.ndarray:
    """Pad to square (keeping aspect) then resize to size×size."""
    h, w = img.shape
    pad = int(max(h, w) * pad_ratio)
    side = max(h, w) + 2 * pad
    canvas = np.full((side, side), bg, np.uint8)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = img
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


# ─── heatmaps for the line / CRAFT dataset ───────────────────────────────────
def gaussian_template(size: int = 64) -> np.ndarray:
    ax = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(ax, ax)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * 0.35 ** 2))
    return (g / g.max()).astype(np.float32)


def place_gaussian(heat: np.ndarray, box, tmpl: np.ndarray):
    x0, y0, x1, y1 = [int(v) for v in box]
    w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    g = cv2.resize(tmpl, (w, h))
    H, W = heat.shape
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x0 + w, W), min(y0 + h, H)
    heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], g[: y1 - y0, : x1 - x0])


# ─── main generation ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--per-class", type=int, help="override glyphs.per_class_per_font")
    ap.add_argument("--lines", type=int, help="override lines.count")
    ap.add_argument("--fonts", type=int, help="use only the first N fonts (quick runs)")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    s = cfg["synth"]
    rng = np.random.default_rng(s["seed"])

    fonts_dir = resolve(ROOT, cfg["paths"]["fonts"])
    out_dir = resolve(ROOT, cfg["paths"]["synth_out"])
    overlays = resolve(ROOT, cfg["paths"]["overlays"])
    out_dir.mkdir(parents=True, exist_ok=True)
    overlays.mkdir(parents=True, exist_ok=True)

    alphabet = cfg["alphabet"]["base"] + cfg["alphabet"]["final"]
    cls_index = {c: i for i, c in enumerate(alphabet)}
    (out_dir / "classes.json").write_text(
        json.dumps({"alphabet": alphabet, "index": cls_index}, ensure_ascii=False, indent=2)
    )

    font_paths = sorted(fonts_dir.glob("*.ttf")) + sorted(fonts_dir.glob("*.otf"))
    if args.fonts:
        font_paths = font_paths[: args.fonts]
    if not font_paths:
        raise SystemExit(f"No fonts found in {fonts_dir}")

    per_class = args.per_class or s["glyphs"]["per_class_per_font"]
    n_lines = args.lines if args.lines is not None else s["lines"]["count"]
    aug = Augmenter(cfg["augment"], rng)
    tmpl = gaussian_template()

    print(f"fonts={len(font_paths)}  per_class={per_class}  lines={n_lines}")

    # ── isolated glyphs ──────────────────────────────────────────────────────
    gpat = str(out_dir / f"{s['packaging']['glyph_shard_prefix']}-%06d.tar")
    sample_tiles, n_glyph = [], 0
    with wds.ShardWriter(gpat, maxcount=s["packaging"]["shard_size"]) as sink:
        for fp in tqdm(font_paths, desc="glyphs"):
            font = ImageFont.truetype(str(fp), s["render_px"])
            for ch in alphabet:
                if not has_glyph(fp, ch):
                    continue
                base = render_glyph(font, ch)
                if base is None:
                    continue
                ink0 = base.astype(np.float32) / 255.0
                for _ in range(per_class):
                    ink = aug.glyph(ink0.copy())
                    raster = compose_on_paper(ink, cfg["augment"], rng)
                    tile = square_fit(raster, s["glyphs"]["out_size"], s["glyphs"]["pad_ratio"])
                    sink.write({
                        "__key__": f"g{n_glyph:08d}",
                        "png": Image.fromarray(tile),
                        "cls": cls_index[ch],
                    })
                    n_glyph += 1
                    if len(sample_tiles) < s["sample_sheet"]["n"] and rng.random() < 0.05:
                        sample_tiles.append(tile)

    # ── multi-letter lines + heatmaps ────────────────────────────────────────
    n_line = 0
    manifest = open(out_dir / s["packaging"]["manifest"], "w")
    if s["lines"]["enabled"] and n_lines > 0:
        lcfg = s["lines"]
        lpat = str(out_dir / f"{s['packaging']['line_shard_prefix']}-%06d.tar")
        # Pre-render a glyph bank per font for speed.
        with wds.ShardWriter(lpat, maxcount=s["packaging"]["shard_size"]) as sink:
            for _ in tqdm(range(n_lines), desc="lines"):
                fp = font_paths[int(rng.integers(len(font_paths)))]
                font = ImageFont.truetype(str(fp), lcfg["canvas_h"] - 24)
                n = int(rng.integers(lcfg["min_chars"], lcfg["max_chars"] + 1))
                chars = [alphabet[int(rng.integers(len(alphabet)))] for _ in range(n)]
                glyphs = []
                for ch in chars:
                    if not has_glyph(fp, ch):
                        continue
                    base = render_glyph(font, ch)
                    if base is None:
                        continue
                    ink = aug.glyph(base.astype(np.float32) / 255.0)
                    glyphs.append((ch, ink))
                if not glyphs:
                    continue
                gap_lo, gap_hi = lcfg["space_px"]
                widths = [g[1].shape[1] for g in glyphs]
                total_w = sum(widths) + int(gap_hi) * (len(glyphs) + 1)
                H = lcfg["canvas_h"]
                ink_canvas = np.zeros((H, total_w), np.float32)
                boxes, labels = [], []
                x = int(rng.integers(gap_lo, gap_hi + 1))
                for ch, ink in glyphs:
                    gh, gw = ink.shape
                    y = int((H - gh) / 2 + rng.integers(-6, 7))
                    y = max(0, min(y, H - gh))
                    ink_canvas[y:y + gh, x:x + gw] = np.maximum(ink_canvas[y:y + gh, x:x + gw], ink)
                    boxes.append([x, y, x + gw, y + gh])
                    labels.append(cls_index[ch])
                    x += gw + int(rng.integers(gap_lo, gap_hi + 1))
                line_img = compose_on_paper(ink_canvas[:, :x], cfg["augment"], rng)
                W = line_img.shape[1]
                region = np.zeros((H, W), np.float32)
                affinity = np.zeros((H, W), np.float32)
                for b in boxes:
                    place_gaussian(region, b, tmpl)
                for a, bb in zip(boxes, boxes[1:]):
                    cx0 = (a[0] + a[2]) / 2; cx1 = (bb[0] + bb[2]) / 2
                    cy0 = (a[1] + a[3]) / 2; cy1 = (bb[1] + bb[3]) / 2
                    bw = abs(cx1 - cx0); bh = (a[3] - a[1] + bb[3] - bb[1]) / 4
                    place_gaussian(affinity,
                                   [min(cx0, cx1), (cy0 + cy1) / 2 - bh,
                                    max(cx0, cx1), (cy0 + cy1) / 2 + bh], tmpl)
                key = f"l{n_line:08d}"
                sink.write({
                    "__key__": key,
                    "png": Image.fromarray(line_img),
                    "region.png": Image.fromarray((region * 255).astype(np.uint8)),
                    "affinity.png": Image.fromarray((affinity * 255).astype(np.uint8)),
                    "json": {"boxes": boxes, "labels": labels},
                })
                manifest.write(json.dumps({"key": key, "boxes": boxes, "labels": labels}) + "\n")
                n_line += 1
    manifest.close()

    # ── sample sheet ─────────────────────────────────────────────────────────
    if sample_tiles:
        n = len(sample_tiles)
        cols = 10
        rows = math.ceil(n / cols)
        sz = sample_tiles[0].shape[0]
        sheet = np.full((rows * sz, cols * sz), 255, np.uint8)
        for i, t in enumerate(sample_tiles):
            r, c = divmod(i, cols)
            sheet[r * sz:(r + 1) * sz, c * sz:(c + 1) * sz] = t
        Image.fromarray(sheet).save(resolve(ROOT, s["sample_sheet"]["path"]))

    print(f"\nDONE  glyphs={n_glyph}  lines={n_line}")
    print(f"  shards + manifest -> {out_dir}")
    print(f"  sample sheet      -> {resolve(ROOT, s['sample_sheet']['path'])}")
    print(f"  classes.json      -> {out_dir / 'classes.json'}")


if __name__ == "__main__":
    main()
