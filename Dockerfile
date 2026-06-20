# Serving image for the Hebrew-handwriting demo (src/app.py — FastAPI on :8000).
# Only the runtime path is installed: CRAFT word detection + the FastAPI web app.
# The CRAFT weights (craft_mlt_25k.pth) are gitignored, so they are not in the
# build context — instead they are downloaded from the official source during the
# build and baked into the image, so the VM gets them automatically on pull.
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

# Download the official CRAFT "General" weights (craft_mlt_25k.pth) and bake them
# in. Default is the canonical Google Drive id from third_party/CRAFT-pytorch's
# README; override with --build-arg CRAFT_GDRIVE_ID=... if you mirror it elsewhere.
ARG CRAFT_GDRIVE_ID=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ
RUN pip install --no-cache-dir gdown \
    && mkdir -p /app/models \
    && gdown "https://drive.google.com/uc?id=${CRAFT_GDRIVE_ID}" \
        -O /app/models/craft_mlt_25k.pth \
    && test -s /app/models/craft_mlt_25k.pth \
    && pip uninstall -y -q gdown

# App code + the CRAFT inference code (third_party is a git submodule).
COPY src/ ./src/
COPY third_party/ ./third_party/
COPY config.yaml ./config.yaml

EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
