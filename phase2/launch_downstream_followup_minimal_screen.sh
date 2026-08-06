#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${1:-ds_followup_minimal}"
MODE="${2:-all}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SCREEN_LOG="${ROOT_DIR}/logs/downstream_followup_minimal_${MODE}_${TIMESTAMP}.log"

cd "${ROOT_DIR}"

if screen -list | grep -q "[.]${SESSION_NAME}[[:space:]]"; then
  echo "screen session already exists: ${SESSION_NAME}" >&2
  exit 1
fi

mkdir -p "${ROOT_DIR}/logs"

screen -L -Logfile "${SCREEN_LOG}" -dmS "${SESSION_NAME}" \
  bash -lc "set -euo pipefail; cd '${ROOT_DIR}' && exec bash phase2/run_downstream_followup_minimal.sh '${MODE}'"

echo "started screen session ${SESSION_NAME}"
echo "screen log: ${SCREEN_LOG}"
