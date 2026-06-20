#!/usr/bin/env bash
# Stop every running container, then bring up ONLY font-extractor (port 80).
# Run on the Azure VM.  ->  http://20.197.16.237/
set -euo pipefail

NAME=font-extractor
IMAGE=cyttic/font-extractor:latest

echo ">> stopping all running containers..."
ids=$(docker ps -q)
if [ -n "$ids" ]; then docker stop $ids; else echo "   (none running)"; fi

echo ">> starting $NAME ..."
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  docker start "$NAME"
else
  # container doesn't exist yet (first run) -> pull the latest image and create it
  docker pull "$IMAGE"
  docker run -d --name "$NAME" --restart unless-stopped --network host \
    -v "$HOME/fe-logs":/app/logs "$IMAGE"
fi

echo ">> status:"
docker ps --filter "name=$NAME" --format '{{.Names}}\t{{.Status}}'
echo ">> open: http://20.197.16.237/"
