#!/usr/bin/env bash
# Print the font-extractor analyze access log (time / ip / device / recognized).
# Runs on the VM — meant to be called by OpenClaw.
#   analyze-log.sh           last 100 lines + summary
#   analyze-log.sh 500       last 500 lines + summary
#   analyze-log.sh all       everything + summary
set -u

N="${1:-100}"
LOG="$HOME/fe-logs/analyze.log"

# Source the log: the persistent file (sudo if it's root-owned), else docker stdout.
read_log() {
  if [ -r "$LOG" ]; then
    cat "$LOG"
  elif [ -f "$LOG" ] && sudo -n true 2>/dev/null; then
    sudo cat "$LOG"
  else
    docker logs font-extractor 2>&1 | grep '^ANALYZE' | sed 's/^ANALYZE //'
  fi
}

DATA="$(read_log)"

echo "=== analyze log ==="
if [ "$N" = "all" ]; then echo "$DATA"; else echo "$DATA" | tail -n "$N"; fi

echo
echo "=== summary ==="
echo "total analyses : $(echo "$DATA" | grep -c 'ip=')"
echo "-- by device --"
echo "$DATA" | grep -oE 'device=[^ ]+' | sort | uniq -c | sort -rn
echo "-- top IPs --"
echo "$DATA" | grep -oE 'ip=[^ ]+' | sort | uniq -c | sort -rn | head
