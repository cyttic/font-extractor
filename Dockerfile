# Serving image for the Hebrew-handwriting demo (src/app.py — FastAPI on :80).
# Only the runtime path is installed: CRAFT word detection + the FastAPI web app.
# The CRAFT weights (craft_mlt_25k.pth) are stored in the repo via Git LFS and
# copied into the image below, so it is self-contained and the VM gets them
# automatically on pull. Build context must contain the real file, so CI checks
# out with lfs: true (see .github/workflows/deploy.yml).
FROM python:3.12-slim

# opencv-python-headless still needs libglib2.0-0 at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch keeps the image small (no CUDA), then the rest of the runtime deps.
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        fastapi "uvicorn[standard]" python-multipart requests \
        opencv-python-headless numpy pillow scikit-image scipy pyyaml

# CRAFT "General" weights — stored in the repo via Git LFS and copied straight in,
# so the image is self-contained with no external download at build time. (CI must
# check out with lfs: true so this is the real file, not a 130-byte LFS pointer.)
COPY models/craft_mlt_25k.pth ./models/craft_mlt_25k.pth

# App code + the vendored CRAFT inference code.
COPY src/ ./src/
COPY third_party/ ./third_party/
COPY config.yaml ./config.yaml

# Serve on port 80 so the bare host URL works (http://<vm-ip>/). Binding the
# privileged port is fine: --network host + the container runs as root.
EXPOSE 80
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "80"]
