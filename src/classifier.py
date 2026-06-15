#!/usr/bin/env python3
"""Phase 4 — 27-class Hebrew letterform classifier (CNN).

Trains on the isolated-glyph WebDataset shards produced by Phase 1. Designed to run
on Kaggle GPU: reads shards from --data (default /kaggle/input/...), writes the model,
metrics and a confusion matrix to --out (default /kaggle/working).

The Kaggle notebook (notebooks/kaggle_classifier.ipynb) mirrors this file inline so it
runs standalone; keep the two in sync.

Local CPU smoke test (needs torch installed):
    python src/classifier.py --data data/synth --out models --epochs 2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np


def load_glyphs(data_dir: str, prefix: str = "glyphs", img: int = 64):
    """Read all isolated-glyph shards into memory -> (X uint8 [N,img,img], y [N], classes)."""
    import webdataset as wds
    shards = sorted(glob.glob(os.path.join(data_dir, f"{prefix}-*.tar")))
    if not shards:
        raise SystemExit(f"No '{prefix}-*.tar' shards in {data_dir}")
    classes = None
    cpath = os.path.join(data_dir, "classes.json")
    if os.path.exists(cpath):
        classes = json.load(open(cpath))["alphabet"]
    X, y = [], []
    ds = wds.WebDataset(shards, shardshuffle=False).decode("pil")
    for s in ds:
        a = np.asarray(s["png"].convert("L"))
        if a.shape != (img, img):
            from PIL import Image
            a = np.asarray(Image.fromarray(a).resize((img, img)))
        X.append(a)
        y.append(int(s["cls"]))
    return np.stack(X), np.asarray(y, np.int64), classes


def build_model(num_classes: int):
    import torch.nn as nn

    def block(ci, co):
        return nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    return nn.Sequential(
        block(1, 32), block(32, 64), block(64, 128),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(128, 256), nn.ReLU(inplace=True), nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/kaggle/input/hebrew-synth/synth")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--img", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset, random_split

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.out).mkdir(parents=True, exist_ok=True)

    X, y, classes = load_glyphs(args.data, img=args.img)
    num_classes = int(y.max()) + 1 if classes is None else len(classes)
    print(f"loaded {len(X)} glyphs, {num_classes} classes, device={device}")

    Xt = torch.from_numpy(X).float().div_(255).unsqueeze(1)
    yt = torch.from_numpy(y)
    full = TensorDataset(Xt, yt)
    n_val = int(len(full) * args.val_split)
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(full, [len(full) - n_val, n_val], generator=g)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch, num_workers=2)

    model = build_model(num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = torch.nn.CrossEntropyLoss()

    best_acc, best_cm = 0.0, None
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        sched.step()

        model.eval()
        correct = 0
        cm = np.zeros((num_classes, num_classes), np.int64)
        with torch.no_grad():
            for xb, yb in val_dl:
                pred = model(xb.to(device)).argmax(1).cpu()
                correct += (pred == yb).sum().item()
                for t, p in zip(yb.numpy(), pred.numpy()):
                    cm[t, p] += 1
        acc = correct / max(len(val_ds), 1)
        print(f"epoch {ep + 1:2d}/{args.epochs}  train_loss={tot / len(train_ds):.4f}  val_acc={acc:.4f}")
        if acc >= best_acc:
            best_acc, best_cm = acc, cm
            torch.save({"state_dict": model.state_dict(),
                        "num_classes": num_classes, "classes": classes, "img": args.img},
                       os.path.join(args.out, "classifier.pth"))

    json.dump({"val_acc": best_acc, "classes": classes},
              open(os.path.join(args.out, "metrics.json"), "w"), ensure_ascii=False, indent=2)
    save_confusion(best_cm, classes, os.path.join(args.out, "confusion_matrix.png"))
    print(f"\nBEST val_acc={best_acc:.4f}  ->  {args.out}/classifier.pth")


def save_confusion(cm, classes, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = cm.shape[0]
        cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.imshow(cmn, cmap="viridis")
        labels = classes or [str(i) for i in range(n)]
        ax.set_xticks(range(n)); ax.set_xticklabels(labels)
        ax.set_yticks(range(n)); ax.set_yticklabels(labels)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title("Confusion matrix (row-normalized)")
        fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    except Exception as e:  # plotting is best-effort
        print("confusion plot skipped:", e)


if __name__ == "__main__":
    main()
