# Running the training on Kaggle (GPU)

GPU training (P4 classifier, and later P2 CRAFT) runs on Kaggle. Data is generated
locally, uploaded once as a Kaggle Dataset, and the notebook streams it from
`/kaggle/input/...`. This file is the end-to-end runbook for the **P4 classifier**.

---

## Step 1 — Generate the synthetic data locally (CPU)

```bash
source .venv/bin/activate
python src/synth_data.py          # full run (uses config.yaml)
```

Produces in `data/synth/`:
- `glyphs-*.tar`  — isolated single-glyph shards (what the classifier trains on)
- `lines-*.tar`   — line images + region/affinity heatmaps (for P2 CRAFT, later)
- `classes.json`  — index ↔ letterform map (must travel with the data)
- `manifest.jsonl`
- `output/overlays/p1_sample_sheet.png` — eyeball this first

Tune volume in `config.yaml` → `synth.glyphs.per_class_per_font` (default 30 ≈ 32k glyphs)
and `synth.lines.count`. Quick test run: `python src/synth_data.py --per-class 4 --lines 30 --fonts 3`.

## Step 2 — Upload `data/synth/` as a Kaggle Dataset

**Web UI:** kaggle.com → Datasets → New Dataset → drag the `data/synth/` folder →
name it `hebrew-synth`.

**Or CLI** (`pip install kaggle`, put your token in `~/.kaggle/kaggle.json`):
```bash
cd data
kaggle datasets create -p synth --dir-mode tar    # first time
# later updates:
kaggle datasets version -p synth -m "more glyphs" --dir-mode tar
```
The classifier shards must end up directly under the dataset (with `classes.json`).
The notebook auto-discovers them under `/kaggle/input/**` if the path differs.

## Step 3 — Run the notebook

1. kaggle.com → Code → **New Notebook** → File → Upload → `notebooks/kaggle_classifier.ipynb`.
2. **Add Data** (right panel) → your `hebrew-synth` dataset.
3. **Settings → Accelerator → GPU** (T4 is plenty for this CNN).
4. Check the `DATA` path in the config cell, then **Run All**.

It prints per-epoch `val_acc`, saves the best model, and renders the confusion matrix.

## Step 4 — Pull the results back

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
