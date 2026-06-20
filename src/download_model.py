"""Download crnn.pth from Google Drive after Kaggle training.

Usage:
    python src/download_model.py
    python src/download_model.py --url "https://drive.google.com/file/d/.../view"
    python src/download_model.py --url "..." --out models/crnn.pth
"""
import argparse
import re
from pathlib import Path

import gdown

# ── paste your link here after training ──────────────────────────────────────
DEFAULT_URL = ""   # e.g. 'https://drive.google.com/file/d/1ABC.../view?usp=sharing'
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "models" / "crnn.pth"


def gdrive_download(url: str, out: Path) -> None:
    m = re.search(r"/d/([\w-]+)|[?&]id=([\w-]+)", url)
    if not m:
        raise ValueError(f"Could not extract file ID from URL: {url}")
    file_id = next(g for g in m.groups() if g)

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading crnn.pth → {out}")
    gdown.download(
        f"https://drive.google.com/uc?id={file_id}",
        str(out),
        quiet=False,
    )
    print(f"Saved to {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL,
                    help="Google Drive share link for crnn.pth")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="Local destination path")
    args = ap.parse_args()

    if not args.url:
        raise SystemExit(
            "No URL provided. Either set DEFAULT_URL in the script "
            "or pass --url 'https://drive.google.com/file/d/.../view'"
        )

    gdrive_download(args.url, Path(args.out))
