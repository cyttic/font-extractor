# Thin serving image for the Hebrew-handwriting demo (src/app.py — FastAPI on :80).
# CRAFT detection and TrOCR both run OFF the VM via reverse SSH tunnels
# (127.0.0.1:9001 CRAFT, 127.0.0.1:8001 TrOCR), so this image carries NO torch and
# NO model weights — it stays small and starts fast. Only image I/O + the web app.
FROM python:3.12-slim

# opencv-python-headless still needs libglib2.0-0 at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
        fastapi "uvicorn[standard]" python-multipart requests \
        opencv-python-headless numpy pillow scikit-image scipy

COPY src/ ./src/
COPY config.yaml ./config.yaml

# Serve on port 80 so the bare host URL works (http://<vm-ip>/). Binding the
# privileged port is fine: --network host + the container runs as root.
EXPOSE 80
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "80"]
