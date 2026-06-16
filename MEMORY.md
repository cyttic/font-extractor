# Memory Index

- [Training on Kaggle GPU](memory/training-on-kaggle-gpu.md) — P2/P4 fine-tuning runs on Kaggle, not local; shapes how P1 output is packaged
- [Fonts location](memory/fonts-location.md) — 40 training fonts live at fontsVisualizer/fonts, point config there
- [Kaggle fonts via gdrive zip](memory/kaggle-fonts-via-gdrive-zip.md) — notebook pulls fonts.zip from Drive by file id; gdown folder download is broken
- [Extraction pipeline CRAFT+GT](memory/extraction-pipeline-craft-gt.md) — pretrained CRAFT + GT alignment + classifier verify; no CRAFT training needed
- [Real handwriting dataset](memory/real-handwriting-dataset.md) — dataset_matan, 5317 lines / 496 writers with GT text
- [Decoding and bootstrap](memory/decoding-and-bootstrap.md) — lexicon decoding works (2.5× word acc); classifier is bottleneck (33% real letter acc); bootstrapping on real crops
