# Serving image for the Hebrew-handwriting demo (src/app.py — FastAPI on :8000).
# Only the runtime path is installed: CRAFT word detection + the FastAPI web app.
# Model weights (models/*.pth) are gitignored, so they are NOT baked in — mount
# them at /app/models at run time (see .github/workflows/deploy.yml).
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

# App code + the CRAFT inference code (third_party is a git submodule).
COPY src/ ./src/
COPY third_party/ ./third_party/
COPY config.yaml ./config.yaml

EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
