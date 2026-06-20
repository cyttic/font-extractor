#!/usr/bin/env bash
# Control the Azure VM's Docker from your local machine — one command.
#
#   ./vm.sh font          run ONLY font-extractor   (:80 -> http://20.197.16.237/)
#   ./vm.sh ocr           run ONLY web-hebrew-ocr    (:80 -> http://20.197.16.237/)
#   ./vm.sh stop          stop all containers
#   ./vm.sh ps            list containers (running + stopped)
#   ./vm.sh logs <name>   follow a container's logs
#
# Override the key/host if needed:  VM_KEY=~/my.pem VM_HOST=azureuser@1.2.3.4 ./vm.sh ps
set -euo pipefail

KEY="${VM_KEY:-$HOME/Downloads/vm-framework_key.pem}"
HOST="${VM_HOST:-azureuser@20.197.16.237}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "$HOST")

case "${1:-ps}" in
  font) "${SSH[@]}" 'bash -s' < "$HERE/run-font-extractor.sh" ;;
  ocr)  "${SSH[@]}" 'bash -s' < "$HERE/run-ocr.sh" ;;
  stop) "${SSH[@]}" 'ids=$(docker ps -q); [ -n "$ids" ] && docker stop $ids || echo "(none running)"' ;;
  ps|status) "${SSH[@]}" 'docker ps -a --format "{{.Names}}\t{{.Status}}"' ;;
  logs) "${SSH[@]}" "docker logs --tail 40 -f ${2:?usage: $0 logs <container>}" ;;
  *) echo "usage: $0 {font|ocr|stop|ps|logs <name>}" >&2; exit 1 ;;
esac
