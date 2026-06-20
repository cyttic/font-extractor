"""Train CRNN on Hebrew word crops from the SCE dataset.

Two data sources are merged automatically:
  1. Words_Dataset/  — 19K pre-cropped JPGs, all labeled via word_class.
  2. Data/Images/ + Data/json_labels/ — TIFF pages; word polygons with
     'transcript' are cropped on-the-fly.

Usage (Kaggle GPU):

    python src/train_crnn.py \\
        --dataset /kaggle/input/sce-dataset/Dataset_Output \\
        --out     /kaggle/working/crnn.pth \\
        --epochs  40 --batch 32 --lr 1e-3 --workers 4

Local CPU quick-test:

    python src/train_crnn.py \\
        --dataset /mnt/ssd2/cyttic/datasets/sce_dataset/Dataset_Output \\
        --out     src/models/crnn.pth \\
        --epochs  3 --batch 4 --workers 0 --max-samples 300
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from crnn import CRNN, BLANK, CHAR2IDX, IDX2CHAR, NUM_CLASSES, encode

# ── constants ────────────────────────────────────────────────────────────────
IMG_H = 64
MIN_W = 32
MAX_W = 640    # word crops are narrow — no need for more


# ── data collection ───────────────────────────────────────────────────────────

def build_class_map(json_dir: Path) -> dict[int, str]:
    """Scan JSON labels → word_class id → most common transcript."""
    class_texts: dict[int, Counter] = defaultdict(Counter)
    for jf in json_dir.glob("*.json"):
        try:
            data = json.load(open(jf, encoding="utf-8"))
        except Exception:
            continue
        for shape in data.get("shapes", []):
            if shape.get("type") != "word":
                continue
            cls = shape.get("word_class")
            txt = shape.get("transcript", "").strip()
            if cls is not None and txt:
                class_texts[cls][txt] += 1
    return {cls: ctr.most_common(1)[0][0] for cls, ctr in class_texts.items()}


def collect_words_dataset(base: Path, class_map: dict[int, str]) -> list:
    """Pre-cropped JPGs from Words_Dataset/ — (path, text) pairs."""
    samples = []
    words_dir = base / "Words_Dataset"
    if not words_dir.exists():
        return samples
    for cls_dir in words_dir.iterdir():
        if not cls_dir.name.isdigit():
            continue
        text = "".join(c for c in class_map.get(int(cls_dir.name), "") if c in CHAR2IDX)
        if not text:
            continue
        for p in cls_dir.glob("*.jpg"):
            samples.append((p, text))
    return samples


def collect_tiff_crops(base: Path) -> list:
    """Word crops cut from TIFF pages using JSON polygon annotations — (tif, bbox, text)."""
    samples = []
    img_dir  = base / "Data" / "Images"
    json_dir = base / "Data" / "json_labels"
    if not img_dir.exists() or not json_dir.exists():
        return samples
    for jf in sorted(json_dir.glob("*.json")):
        tif = img_dir / f"{jf.stem}.tif"
        if not tif.exists():
            tif = img_dir / f"{jf.stem}.tiff"
        if not tif.exists():
            continue
        try:
            data = json.load(open(jf, encoding="utf-8"))
        except Exception:
            continue
        for shape in data.get("shapes", []):
            if shape.get("type") != "word":
                continue
            txt = "".join(c for c in shape.get("transcript", "") if c in CHAR2IDX)
            if not txt:
                continue
            pts = np.array(shape["points"])
            x0, y0 = int(pts[:, 0].min()), int(pts[:, 1].min())
            x1, y1 = int(pts[:, 0].max()), int(pts[:, 1].max())
            if x1 > x0 and y1 > y0:
                samples.append((tif, (x0, y0, x1, y1), txt))
    return samples


# ── image helpers ─────────────────────────────────────────────────────────────

def load_gray(path: Path) -> np.ndarray | None:
    path = str(path)
    # TIFFs: skip OpenCV entirely (it can't decode old-JPEG TIFFs and spams
    # stderr at the C/libtiff level). Pillow handles them cleanly.
    if path.lower().endswith((".tif", ".tiff")):
        try:
            from PIL import Image
            return np.array(Image.open(path).convert("L"))
        except Exception:
            try:
                import tifffile
                img = tifffile.imread(path)
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                return img
            except Exception:
                return None
    # Everything else (jpg/png): OpenCV is fastest
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def resize_to_height(img: np.ndarray, h: int = IMG_H) -> np.ndarray:
    oh, ow = img.shape[:2]
    new_w = max(MIN_W, min(MAX_W, int(round(ow * h / oh))))
    return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA)


def augment(img: np.ndarray) -> np.ndarray:
    alpha = random.uniform(0.8, 1.2)
    beta  = random.uniform(-20, 20)
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if random.random() < 0.5:
        noise = np.random.normal(0, random.uniform(2, 8), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.5:
        scale = random.uniform(0.85, 1.15)
        new_w = max(MIN_W, min(MAX_W, int(img.shape[1] * scale)))
        img = cv2.resize(img, (new_w, img.shape[0]), interpolation=cv2.INTER_LINEAR)
    if random.random() < 0.3:
        k = np.ones((2, 2), np.uint8)
        img = cv2.dilate(img, k) if random.random() < 0.5 else cv2.erode(img, k)
    return img


# ── dataset ───────────────────────────────────────────────────────────────────

class HebrewWordDataset(Dataset):
    def __init__(self, samples: list, train: bool = True):
        self.samples = samples
        self.train   = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        if len(item) == 2:
            path, text = item
            img = load_gray(path)
        else:
            tif_path, (x0, y0, x1, y1), text = item
            full = load_gray(tif_path)
            img  = full[y0:y1, x0:x1] if full is not None else None

        if img is None or img.size == 0:
            img  = np.full((IMG_H, MIN_W), 255, np.uint8)
            text = text[:1] or "א"

        img = resize_to_height(img)
        if self.train:
            img = augment(img)

        # Hebrew RTL → reverse to match left-to-right scan order
        labels = encode(text[::-1]) or [1]
        tensor = torch.from_numpy(img).float().div_(255).unsqueeze(0)
        return tensor, labels, img.shape[1]


def collate_fn(batch):
    imgs, labels_list, widths = zip(*batch)
    max_w  = max(img.shape[2] for img in imgs)
    padded = torch.full((len(imgs), 1, IMG_H, max_w), 1.0)
    for i, img in enumerate(imgs):
        padded[i, :, :, :img.shape[2]] = img
    input_lengths  = torch.tensor([w // 2 for w in widths],              dtype=torch.long)
    targets        = torch.tensor([l for ls in labels_list for l in ls], dtype=torch.long)
    target_lengths = torch.tensor([len(ls) for ls in labels_list],       dtype=torch.long)
    return padded, targets, input_lengths, target_lengths


# ── metrics ───────────────────────────────────────────────────────────────────

def edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = tmp
    return dp[n]


# ── training loop ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train(train)
    total_loss = total_ed = total_len = n = 0

    with torch.set_grad_enabled(train):
        for imgs, targets, input_lengths, target_lengths in loader:
            imgs    = imgs.to(device)
            targets = targets.to(device)
            log_probs = model(imgs)
            T = log_probs.shape[0]
            ilens = input_lengths.clamp(max=T).to(device)
            loss  = criterion(log_probs, targets, ilens, target_lengths)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            B = imgs.shape[0]
            total_loss += loss.item() * B
            n          += B

            from crnn import decode
            preds_idx = log_probs.argmax(2).permute(1, 0)
            offset = 0
            for i, tlen in enumerate(target_lengths.tolist()):
                gt   = "".join(IDX2CHAR.get(x, "?")
                               for x in targets[offset:offset+tlen].tolist())[::-1]
                pred = decode(preds_idx[i].tolist())[::-1]
                total_ed  += edit_distance(pred, gt)
                total_len += max(len(gt), 1)
                offset    += tlen

    return total_loss / max(n, 1), total_ed / max(total_len, 1)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",     default="/mnt/ssd2/cyttic/datasets/sce_dataset/Dataset_Output")
    ap.add_argument("--out",         default="src/models/crnn.pth")
    ap.add_argument("--epochs",      type=int,   default=40)
    ap.add_argument("--batch",       type=int,   default=32)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--hidden",      type=int,   default=256)
    ap.add_argument("--workers",     type=int,   default=4)
    ap.add_argument("--val-split",   type=float, default=0.1)
    ap.add_argument("--max-samples", type=int,   default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    base     = Path(args.dataset)
    json_dir = base / "Data" / "json_labels"

    print("Building class map …")
    class_map = build_class_map(json_dir)
    print(f"  {len(class_map)} word classes resolved")

    print("Collecting samples …")
    samples = collect_words_dataset(base, class_map) + collect_tiff_crops(base)
    random.shuffle(samples)
    print(f"  Total: {len(samples)}")

    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    n_val   = max(1, int(len(samples) * args.val_split))
    train_ds = HebrewWordDataset(samples[:-n_val], train=True)
    val_ds   = HebrewWordDataset(samples[-n_val:], train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.workers,
                              pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              collate_fn=collate_fn, num_workers=args.workers,
                              pin_memory=(device.type == "cuda"))

    model     = CRNN(num_classes=NUM_CLASSES, hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CTCLoss(blank=BLANK, reduction="mean", zero_infinity=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_cer = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_cer = run_epoch(model, train_loader, optimizer, criterion, device, True)
        vl_loss, vl_cer = run_epoch(model, val_loader,   optimizer, criterion, device, False)
        scheduler.step()
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train loss={tr_loss:.4f} CER={tr_cer:.3f}  "
              f"val loss={vl_loss:.4f} CER={vl_cer:.3f}")
        if vl_cer < best_cer:
            best_cer = vl_cer
            torch.save({"model_state": model.state_dict(), "num_classes": NUM_CLASSES,
                        "hidden": args.hidden, "epoch": epoch, "val_cer": vl_cer}, out_path)
            print(f"  ✓ saved  best val CER={best_cer:.3f}")

    print("Done.")


if __name__ == "__main__":
    main()
