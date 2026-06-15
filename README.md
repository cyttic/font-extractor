# Hebrew Handwriting → TTF Font Generator

Turn scanned **handwritten Hebrew** (casual *ktav yad*, no nikud, separated letters)
into a usable **TTF font** with all **27 letterforms** (22 base + 5 final).

Final glyphs come **only from the user's real handwriting**. The 40 supplied fonts are
training/scaffolding data only — they never appear in the output font.

## Split: local CPU vs Kaggle GPU

| Stage | Where | Notes |
|-------|-------|-------|
| P1 synth data | **local (CPU)** | renders the 40 fonts → sharded datasets |
| P2 CRAFT fine-tune *(optional)* | **Kaggle GPU** | only if P3 fails on real scans |
| P3 detection | local (CPU) | connected-components default |
| P4 classifier | **Kaggle GPU** | 27-class CNN |
| P5–P8 label / review / vectorize / build | local | FontForge + potrace |

## Setup (local)

System packages (not pip):
```bash
sudo apt install fontforge potrace      # already have fontforge here
```

Python (Homebrew Python 3.14 is externally managed → use a venv):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Kaggle

torch/opencv/numpy are preinstalled on Kaggle GPU images. In the notebook:
```bash
pip install -r requirements-kaggle.txt
```
Phase 1 output is packaged as **tar/WebDataset shards** + a `manifest.jsonl`, uploaded as
a Kaggle Dataset, and streamed from `/kaggle/input/...`. Checkpoints go to `/kaggle/working/`.

## Config

Everything tunable lives in `config.yaml` — paths, alphabet, P1 render/augmentation
thresholds, and per-phase hyperparameters. The 40 fonts are referenced by absolute path
(`/mnt/ssd2/cyttic/projects/fontsVisualizer/fonts`), not copied into the repo.

## Build order

`P1 → P3 → P4 → P5 → P6 → P7 → P8` (the working spine). Add P2 only if P3 underperforms
on real scans. Stop and verify acceptance criteria after each phase. See the
[build spec](hebrew_handwriting_font_generator_SPEC.md).
