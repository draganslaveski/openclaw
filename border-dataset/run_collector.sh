#!/usr/bin/env bash
set -euo pipefail

LOCK_FILE="/tmp/border-dataset-collector.lock"
SCRIPT="/home/dragan-slaveski/.openclaw/workspace/border-dataset/collect_samples.py"
PYTHON_BIN="/home/dragan-slaveski/.openclaw/.venv/bin/python"
LOG_FILE="/home/dragan-slaveski/.openclaw/workspace/border-dataset/logs/collector.log"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] SKIP already running" >> "$LOG_FILE"
  exit 0
fi

{
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] START"
  "$PYTHON_BIN" "$SCRIPT"
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] END ok"
} >> "$LOG_FILE" 2>&1
