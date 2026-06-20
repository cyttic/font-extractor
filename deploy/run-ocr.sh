#!/usr/bin/env bash
# Stop every running container, then bring up ONLY web-hebrew-ocr (port 80).
# Run on the Azure VM.  ->  http://20.197.16.237/
set -euo pipefail

NAME=web-hebrew-ocr

echo ">> stopping all running containers..."
ids=$(docker ps -q)
if [ -n "$ids" ]; then docker stop $ids; else echo "   (none running)"; fi

echo ">> starting $NAME ..."
docker start "$NAME"

echo ">> status:"
docker ps --filter "name=$NAME" --format '{{.Names}}\t{{.Status}}'
echo ">> open: http://20.197.16.237/"
