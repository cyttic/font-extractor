# Build Spec — Hebrew Handwriting → TTF Font Generator

> **For the coding agent:** This is a phased build brief. Build **one phase at a time**, in order. Each phase has a **Deliverable** and **Acceptance criteria** — do not move to the next phase until the current one passes its acceptance criteria. Ask the user for missing inputs (scans, fonts) rather than fabricating them. Keep every stage **inspectable**: write intermediate artifacts (crops, overlays, heatmaps) to disk so a human can eyeball them.

---

## 1. Goal

Turn scanned **handwritten Hebrew** (casual script / *ktav yad*, **no nikud**, letters generally **separated**) into a usable **TTF font** containing all **27 Hebrew letterforms**.

The final font's glyphs come **only from the user's real scanned handwriting**. The 40 supplied fonts are used **as training/scaffolding data for the models** — they must **never** appear in the output font.

### Target alphabet (27 letterforms)
- **22 base:** א ב ג ד ה ו ז ח ט י כ ל מ נ ס ע פ צ ק ר ש ת
- **5 final (sofit):** ך ם ן ף ץ

---

## 2. Inputs (ask the user to provide)

1. `fonts/` — ~40 casual Hebrew handwriting-style fonts (`.ttf`/`.otf`). Used for synthetic data only.
2. `scans/` — the real handwritten Hebrew lists (images). Source of the final glyphs.
3. (Optional) a GPU environment for Phase 2/4 training. CPU is fine for everything else.

---

## 3. Architecture (data flow)

```
                 ┌─────────────────── 40 fonts (training data only) ───────────────────┐
                 │                                                                      │
                 ▼                                                                      ▼
   [P1] Synthetic data generator                                          [P4] Letter classifier (CNN, 27 classes)
   render Hebrew + char boxes + CRAFT heatmaps + augmentation                    │
                 │                                                                │
                 ▼                                                                │
   [P2] Fine-tune CRAFT on Hebrew  ──────────┐                                   │
   (optional / for messy pages)              │                                   │
                                             ▼                                   ▼
 real scans ─► [P3] Detection ─► crops ─► [P5] Label crops ─► [P6] Human review/pick ─► best instance per letter
              (CC default, CRAFT fallback)                                                        │
                                                                                                  ▼
                                                              [P7] Vectorize (potrace) ─► [P8] Assemble in FontForge ─► TTF
```

**Two valid detection paths.** Default to deterministic connected-components (P3). CRAFT (P2) is the fallback/robustness path for degraded pages. Build P3 first; build P2 only if P3 visibly fails on the user's real scans.

---

## 4. Tech stack

- **Language:** Python 3.10+
- **CV / imaging:** OpenCV, NumPy, scikit-image, Pillow
- **Font rendering (synthetic data):** Pillow `ImageFont` + HarfBuzz (`uharfbuzz`) for correct Hebrew shaping, or `fonttools`/`freetype-py` to pull exact glyph metrics
- **Classifier:** PyTorch (small CNN)
- **CRAFT training (optional):** a community training reimplementation (the official `clovaai/CRAFT-pytorch` ships **inference only** — no training code). Use **`backtime92/CRAFT-Reimplementation`** or equivalent, initialized from the official `craft_mlt_25k.pth` weights.
- **Vectorization:** `potrace` (CLI) or `pypotrace`; FontForge's autotrace is an acceptable alternative
- **Font assembly:** **FontForge** Python module (`import fontforge`) — primary. `fonttools` as alternative for low-level table edits.

Provide a `requirements.txt` and a `README.md` with setup steps. Note that FontForge and potrace are system packages, not pip-only — document the install (`apt install fontforge potrace` / brew equivalents).

---

## 5. Repo layout

```
hebrew-font/
├── README.md
├── requirements.txt
├── config.yaml                 # all thresholds/paths in one place
├── data/
│   ├── fonts/                  # user-supplied
│   ├── scans/                  # user-supplied
│   ├── synth/                  # generated (P1)
│   └── crops/                  # detected crops (P3)
├── src/
│   ├── synth_data.py           # P1
│   ├── train_craft.py          # P2 (optional)
│   ├── detect.py               # P3  (cc + craft backends)
│   ├── classifier.py           # P4 train
│   ├── label_crops.py          # P5
│   ├── review_app.py           # P6  (human-in-the-loop UI)
│   ├── vectorize.py            # P7
│   └── build_font.py           # P8
├── models/                     # saved weights
└── output/
    ├── overlays/               # debug visualizations
    └── MyHandwriting.ttf       # final
```

Put **every tunable threshold in `config.yaml`** (binarization window, min/max component size, split-width ratio, merge-gap, em size, baseline). No magic numbers hardcoded in logic.

---

## Phase 1 — Synthetic data generator (from the 40 fonts)

**Why first:** it produces the labeled data both the classifier (P4) and CRAFT (P2) depend on. It's also the highest-leverage piece.

**Build `src/synth_data.py` to:**
1. For each font, render Hebrew strings (random sequences of the 27 letterforms + spaces) onto a canvas. Use HarfBuzz shaping so glyphs render correctly; render **RTL** but note reading order is irrelevant downstream.
2. Record the **exact bounding box of every rendered glyph** (you placed it, so you know it — this is free perfect ground truth, the SynthText trick).
3. Emit two output formats:
   - **Isolated single-glyph images** labeled by letterform → for the classifier (P4).
   - **Multi-letter line images + per-character boxes** → for CRAFT (P2), converted to CRAFT's two **Gaussian heatmap targets**: a **region score** (Gaussian at each char center) and an **affinity score** (Gaussian centered between adjacent char centers, derived from the char quadrilaterals).
4. **Augmentation pipeline (mandatory — fonts have zero intra-class variation):** elastic/grid distortion, per-glyph rotation & slant jitter, random spacing, stroke thicken/thin (morphological erode/dilate), Gaussian blur, additive noise, paper/background textures, light ink-bleed simulation. Augmentation must be reproducible (seeded) and toggleable per-run.

**Deliverable:** `data/synth/` populated with both dataset formats + a sample sheet of ~50 augmented examples written to `output/overlays/`.

**Acceptance criteria:**
- Box overlays on rendered lines are visually correct (boxes hug glyphs) for a sampled set.
- All 27 letterforms appear, including the 5 finals, each in its own class.
- Augmented samples look plausibly hand-like, not destroyed.

---

## Phase 2 — Fine-tune CRAFT on Hebrew *(optional — build only if P3 fails on real scans)*

**Requires a CUDA GPU** (~8 GB floor; 12–16 GB comfortable). Colab/Kaggle/cloud rental is fine; CPU training is not viable.

**Build `src/train_craft.py` to:**
1. Use a CRAFT **training reimplementation** (official repo has no training code). Document exactly which repo/commit.
2. Initialize from official `craft_mlt_25k.pth` (fine-tune, don't train from scratch — this is what makes ~40 fonts enough).
3. Feed the P1 line-images + heatmap targets. Train with the standard CRAFT MSE loss on region + affinity maps (with OHEM if the repo supports it).
4. Save best weights to `models/craft_hebrew.pth`. Log train/val loss.

**Acceptance criteria:**
- On a held-out synthetic val set, predicted region/affinity heatmaps visibly localize characters.
- On a few **real** scan samples, character boxes are tighter/more complete than off-the-shelf CRAFT. (If not better than P3 on real scans, **stop — don't use CRAFT**, the deterministic path wins.)

---

## Phase 3 — Detection on real scans (BUILD THIS FIRST after P1)

**Build `src/detect.py` with two selectable backends; default = `cc`.**

### Backend A — `cc` (connected components, deterministic, default)
1. **Binarize** with an adaptive/local method (**Sauvola** preferred over global Otsu for uneven handwriting scans).
2. **Deskew** (estimate dominant text angle, rotate).
3. **Line split** via horizontal projection profile (valleys = line breaks).
4. **Connected-component analysis** per line.
5. **Cleanup pass — the load-bearing logic. Two opposing fixes:**
   - **MERGE** components that are pieces of one letter. Hebrew multi-stroke letters over-segment: **ה ק א ש ת** especially (detached legs/feet/strokes). Merge by horizontal overlap + small vertical/intra-band gaps.
   - **SPLIT** blobs that are two touching letters: flag any component much wider than the line's **median letter width**, cut at the deepest valley in its **vertical projection**.
6. **Noise filter:** filter by **size AND density combined**, never size alone — protect genuinely small letters (**י** especially; thin **ו ז ן ר**).

### Backend B — `craft` (fallback)
- Load `models/craft_hebrew.pth` (P2) or official weights; run inference; emit char boxes. **Still run the P5 merge/split-aware labeling afterward** — CRAFT does not eliminate the Hebrew cleanup problem, it only reduces it.

**Both backends output:** cropped character images to `data/crops/` (full ink, no clipping) + an **overlay image per scan** (boxes drawn on the original) to `output/overlays/`.

**Acceptance criteria:**
- Overlays show boxes that capture **complete** letters without clipping strokes.
- Multi-stroke letters (ה ק א ש ת) are merged into one box, not several.
- Touching-letter blobs are split.
- At least one clean instance of each of the 27 letterforms is recoverable across the scans (this is the real bar — not per-letter perfection, just ≥1 good copy each).

---

## Phase 4 — Letter classifier (27-class CNN)

**Build `src/classifier.py` to:**
1. Train a small CNN (input ~64×64 grayscale, 27 output classes) on the **isolated-glyph dataset from P1** (with augmentation).
2. Save to `models/classifier.pth` + report val accuracy and a confusion matrix.

**Note:** This is the **only genuinely "model" component** that's clearly worth it, and the 40 fonts map onto it perfectly. Do **not** use Tesseract for this — its Hebrew accuracy is weak, and you only need a 27-way bucketer.

**Acceptance criteria:**
- Val accuracy high on synthetic data; confusion matrix highlights expected look-alikes (e.g. ב/כ, ד/ר, ה/ח/ת, ם/ס, final vs base forms) so the human reviewer knows where to look.

---

## Phase 5 — Label the real crops

**Build `src/label_crops.py` to:**
1. Run the P4 classifier on every crop in `data/crops/`.
2. Bucket crops by predicted letterform; store each crop's softmax confidence.
3. Write a per-letter gallery (crops sorted by confidence) for the reviewer.

**Acceptance criteria:** every crop assigned a label + confidence; galleries written; obvious mislabels are flagged low-confidence (not silently confident).

---

## Phase 6 — Human-in-the-loop review & glyph selection (do not skip)

**Build `src/review_app.py` — a minimal local UI (e.g. Streamlit/Flask).** Since only ~27 letterforms are needed once, a few minutes of human clicking beats chasing full automation.

**Features:**
- Show, per letterform, the candidate crops (with the scan overlay context).
- Let the user: **reassign** a wrong label, **fix/redraw** a bounding box, **approve** the single best instance per letter.
- Export the 27 approved glyph images to `data/crops/approved/`.

**Acceptance criteria:** user can produce exactly one approved, clean, complete image per letterform for all 27.

---

## Phase 7 — Vectorize approved glyphs

**Build `src/vectorize.py` to:**
1. Clean each approved raster glyph (denoise, optional smoothing, ensure clean black-on-white bitmap).
2. Trace raster → vector outline with **potrace** (tune turdsize/alphamax/opttolerance) → SVG/EPS or path data FontForge can import.

**Acceptance criteria:** vector outlines are smooth, closed, and faithful to the handwriting (no jagged stair-steps, no dropped counters/holes in ב ם ס ע etc.).

---

## Phase 8 — Assemble the TTF (metrics are where this lives or dies)

**Build `src/build_font.py` using the FontForge Python module to:**
1. Create a font; set em size (1000 or 2048 units), ascent/descent, family/style names, and **Unicode mappings** for each Hebrew letterform (base block U+05D0–U+05EA, including the final-form codepoints).
2. Import each vectorized glyph into its slot.
3. **Set metrics per glyph — the make-or-break step:**
   - Consistent **baseline** alignment.
   - Correct handling of letters that drop **below baseline** (finals ך ן ף ץ, and ק) and rise **above** (ל).
   - Per-glyph **left/right side bearings** and **advance width** so spacing reads evenly (no ransom-note bounce).
4. Validate and **export `output/MyHandwriting.ttf`**.

**Acceptance criteria:**
- TTF installs and renders in a normal text app.
- Typing the 27 letters produces even baseline, even spacing, no clipping, correct below/above-baseline behavior.
- Provide a rendered specimen image (all 27 letters + a sample Hebrew sentence) to `output/`.

---

## 6. Global principles for the agent

- **Build P1 → P3 → P4 → P5 → P6 → P7 → P8 first** (the working spine). Add **P2 only if** P3 underperforms on the real scans.
- **Inspectability over cleverness:** always write overlays/intermediates to disk; prefer rules a human can read and tune over opaque steps.
- **The 40 fonts are scaffolding, not output.** Never let a rendered glyph reach the final TTF.
- **Don't chase perfect automation.** The target is "≥1 clean instance per letter, then human-approved," not "segment every character flawlessly."
- Centralize thresholds in `config.yaml`; keep augmentation seeded/reproducible.
- After each phase, **stop and report** the deliverable + whether acceptance criteria passed before proceeding.
