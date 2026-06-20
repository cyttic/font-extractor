"""CRNN (CNN + BiLSTM + CTC) for Hebrew word recognition.

CNN collapses image height to 1 and halves the width (stride-2), producing a
time sequence that the BiLSTM reads. Output: per-timestep log-probs over the
Hebrew alphabet + CTC blank.

CTC convention used throughout:
  blank = index 0
  Hebrew letters = indices 1..27
"""
import torch
import torch.nn as nn

ALPHABET   = list("אבגדהוזחטיכלמנסעפצקרשת") + list("ךםןףץ")  # 27 chars
BLANK      = 0
CHAR2IDX   = {c: i + 1 for i, c in enumerate(ALPHABET)}
IDX2CHAR   = {i + 1: c for i, c in enumerate(ALPHABET)}
NUM_CLASSES = len(ALPHABET)   # 27; CTC output dim = NUM_CLASSES + 1


def encode(text: str) -> list[int]:
    """Hebrew string → list of character indices (unknown chars skipped)."""
    return [CHAR2IDX[c] for c in text if c in CHAR2IDX]


def decode(indices) -> str:
    """CTC greedy decode: collapse consecutive repeats, remove blanks."""
    out, prev = [], None
    for i in indices:
        i = int(i)
        if i != BLANK and i != prev:
            out.append(IDX2CHAR.get(i, "?"))
        prev = i
    return "".join(out)


class CRNN(nn.Module):
    """CNN backbone + LOCAL 1D-conv head + linear projection.

    NOTE: a BiLSTM head gives great recognition but useless localization —
    with full-sequence context, CTC emits all characters bunched at one edge
    of the image (it only has to get the sequence right, not the positions).
    For character localization we instead use a stack of 1D convolutions with
    a small receptive field, so each output timestep depends only on a local
    image patch and is forced to fire where the letter actually is.

    Input : (B, 1, H=64, W)  — grayscale word image, W variable
    Output: (T, B, NUM_CLASSES+1) log-softmax  where T ≈ W // 2
    """

    def __init__(self, num_classes: int = NUM_CLASSES, hidden: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            # Block 1 — H/2, W/2
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # Block 2 — H/4  (height-only pool, width stays at W/2)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d((2, 1)),
            # Block 3 — H/8
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 1)),
            # Block 4 — H/16
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 1)),
            # Collapse any remaining height to 1 regardless of input H
            nn.AdaptiveAvgPool2d((1, None)),
        )
        # Local-context head over the width sequence (kernel 5 → small RF).
        self.head = nn.Sequential(
            nn.Conv1d(256, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden), nn.ReLU(True),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden), nn.ReLU(True),
        )
        self.fc = nn.Conv1d(hidden, num_classes + 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)            # (B, 256, 1, W')
        feat = feat.squeeze(2)        # (B, 256, W')
        feat = self.head(feat)        # (B, hidden, W')
        feat = self.fc(feat)          # (B, C, W')
        feat = feat.permute(2, 0, 1)  # (W', B, C)
        return feat.log_softmax(2)
