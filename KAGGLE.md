# Running the training on Kaggle (GPU)

The notebook (`notebooks/kaggle_classifier.ipynb`) **downloads the fonts from Google Drive,
generates the synthetic glyph data on Kaggle itself**, then trains the P4 classifier — all
in one run. No Kaggle dataset upload needed.

---

## Step 1 — Share the Drive font folder

The notebook pulls fonts from:
`https://drive.google.com/drive/folders/1D7gY8pb9w9HrfbmFEuWfcaICKGTECpaM`

Make sure that folder is shared **"Anyone with the link"** (Share → General access → Anyone
with the link → Viewer), or `gdown` can't fetch it.

## Step 2 — Run the notebook

1. kaggle.com → Code → **New Notebook** → File → Upload → `notebooks/kaggle_classifier.ipynb`.
2. **Settings → Internet → ON** (required for gdown to reach Drive).
3. **Settings → Accelerator → GPU** (T4 is plenty).
4. **Run All**.

The notebook auto-finds the fonts under `/kaggle/input/**/*.ttf`, renders ~32k augmented
glyphs in memory (cell 3, tune `PER_CLASS`), shows a sample row, trains, prints per-epoch
`val_acc`, and saves the best model + confusion matrix.

> The local `src/synth_data.py` is still the full Phase-1 generator (it also makes the
> `lines-*.tar` CRAFT heatmap data for the optional P2). The notebook only inlines the
> *glyph* subset it needs for the classifier.

## Step 3 — Pull the results back

From the notebook's **Output** tab (or `kaggle kernels output ...`), download into `models/`:
- `classifier.pth`        → used by Phase 5 (`label_crops.py`) locally
- `metrics.json`          → val accuracy
- `confusion_matrix.png`  → check expected look-alikes (ב/כ, ד/ר, ה/ח/ת, ם/ס, base vs final)

```bash
mkdir -p models
# drop the downloaded classifier.pth into models/
```

---

## Notes

- `src/classifier.py` is the canonical training code; the notebook mirrors it inline so
  it runs standalone on Kaggle (no repo clone needed). Keep them in sync if you edit.
- P2 CRAFT (the `lines-*.tar` heatmap data) is **optional** — build it only if P3
  connected-components detection underperforms on your real scans. Its Kaggle notebook
  is not built yet; we'll add it then, using a CRAFT training reimplementation
  (`backtime92/CRAFT-Reimplementation`) initialized from `craft_mlt_25k.pth`.
- Acceptance bar for P4: high synthetic val accuracy + a confusion matrix that lights up
  the expected look-alike pairs (so the human reviewer in P6 knows where to look).
