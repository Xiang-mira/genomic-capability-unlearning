#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${ROOT_DIR}/data/phase2/downstream_reaudit/downstream_reaudit_eval_manifest.csv"
OUT_DIR="${ROOT_DIR}/data/phase2/downstream_followup_minimal"
DEVICE="${DEVICE:-cuda:0}"
CPU_THREADS="${CPU_THREADS:-16}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TASKS="hvue_human_host_tropism,hvue_human_virus_pathogenicity_cini,gue_mouse_3,virobench_dna_taxon_genus"

CHECKPOINTS=(
  "base|"
  "gd_random_control|${ROOT_DIR}/data/phase2/checkpoints_tuned/refseq_gd_projinit_random_ar5_s1000/weights.safetensors"
  "projection_rank32|${ROOT_DIR}/data/phase2/checkpoints_projection_adaptive_rank32/projopt_host5_9_coro0_10_adaptive_basis_rank32/weights.safetensors"
  "gd_loc_s1000|${ROOT_DIR}/data/phase2/checkpoints_tuned/refseq_gd_projinit_loc_ar5_s1000/weights.safetensors"
)

usage() {
  cat <<'EOF'
Usage:
  bash phase2/run_downstream_followup_minimal.sh seed45
  bash phase2/run_downstream_followup_minimal.sh seed46
  bash phase2/run_downstream_followup_minimal.sh all

Behavior:
  seed45: run the minimal follow-up set for seed 45 only
  seed46: run the minimal follow-up set for seed 46 only
  all:    run seed45, then seed46 immediately (only use this if you do NOT want a manual stop gate)

Manual stop gate after seed45:
  Stop and do NOT run seed46 if either candidate still fails:
    1. projection_rank32 does not beat gd_random_control on mean primary-target drop
    2. projection_rank32 still shows split target direction (one primary target down, one up/flat)
    3. gd_loc_s1000 still shows weak target drop or obvious retain damage
EOF
}

activate_env() {
  set +u
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate UT-p1
  set -u
}

validate_inputs() {
  if [[ ! -f "${MANIFEST}" ]]; then
    echo "missing manifest: ${MANIFEST}" >&2
    exit 1
  fi

  local entry checkpoint_name checkpoint_path
  for entry in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r checkpoint_name checkpoint_path <<< "${entry}"
    if [[ -n "${checkpoint_path}" && ! -f "${checkpoint_path}" ]]; then
      echo "missing checkpoint for ${checkpoint_name}: ${checkpoint_path}" >&2
      exit 1
    fi
  done
}

run_one() {
  local checkpoint_name="$1"
  local checkpoint_path="$2"
  local seed="$3"
  local out_dir="${OUT_DIR}/global_host_tropism/${checkpoint_name}/seed_${seed}"

  mkdir -p "${out_dir}"
  local cmd=(
    "${PYTHON_BIN}" -u phase2/eval_benchmarks.py
    --benchmark-manifest "${MANIFEST}"
    --benchmark-scope all
    --task-filter "${TASKS}"
    --out-dir "${out_dir}"
    --seed "${seed}"
    --epochs 3
    --max-steps 600
    --eval-every 200
    --validation-max-rows 1000
    --lora-rank 8
    --lora-alpha 16
    --lora-dropout 0.0
    --train-batch-size 1
    --eval-batch-size 1
    --max-length 512
    --device "${DEVICE}"
    --cpu-threads "${CPU_THREADS}"
    --discard-task-checkpoint
    --resume
  )
  if [[ -n "${checkpoint_path}" ]]; then
    cmd=( "${cmd[@]:0:3}" --ckpt "${checkpoint_path}" "${cmd[@]:3}" )
  fi

  echo "[followup] seed=${seed} checkpoint=${checkpoint_name}"
  echo "[followup] command: ${cmd[*]}"
  (cd "${ROOT_DIR}" && "${cmd[@]}")
}

run_seed() {
  local seed="$1"
  for entry in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r checkpoint_name checkpoint_path <<< "${entry}"
    run_one "${checkpoint_name}" "${checkpoint_path}" "${seed}"
  done
}

write_readme() {
  cat > "${OUT_DIR}/README_followup.txt" <<EOF
Minimal downstream follow-up for projection_rank32 vs gd_loc_s1000 vs gd_random_control.

Tasks:
  ${TASKS//,/ }

Checkpoint order:
  base
  gd_random_control
  projection_rank32
  gd_loc_s1000

Recommended order:
  1. Run seed45
  2. Inspect results under ${OUT_DIR}
  3. Only run seed46 if seed45 still leaves a plausible candidate
EOF
}

main() {
  local mode="${1:-}"
  if [[ -z "${mode}" ]]; then
    usage
    exit 1
  fi

  activate_env
  validate_inputs
  mkdir -p "${OUT_DIR}"
  write_readme

  case "${mode}" in
    seed45)
      run_seed 45
      ;;
    seed46)
      run_seed 46
      ;;
    all)
      run_seed 45
      run_seed 46
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
