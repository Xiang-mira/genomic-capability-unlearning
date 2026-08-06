#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${1:-ds_reaudit_triage}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SCREEN_LOG="${ROOT_DIR}/logs/downstream_reaudit_triage_screen_${TIMESTAMP}.log"

cd "${ROOT_DIR}"

if screen -list | grep -q "[.]${SESSION_NAME}[[:space:]]"; then
  echo "screen session already exists: ${SESSION_NAME}" >&2
  exit 1
fi

mkdir -p "${ROOT_DIR}/logs"

screen -L -Logfile "${SCREEN_LOG}" -dmS "${SESSION_NAME}" \
  bash -lc "set -euo pipefail; set +u; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate UT-p1; set -u; cd '${ROOT_DIR}' && exec python -u phase2/run_downstream_reaudit_triage.py"

echo "started screen session ${SESSION_NAME}"
echo "screen log: ${SCREEN_LOG}"
