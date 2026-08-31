#!/usr/bin/env bash
# Phase 2 orchestration: splits -> unlearning runs -> per-run evaluation.
# Usage:
#   bash phase2/run.sh splits              # one-time: build forget/retain CSVs
#   bash phase2/run.sh audit               # dataset / manifest / checkpoint availability audit
#   bash phase2/run.sh prepare_benchmarks  # rebuild benchmark manifest from raw HVUE/GUE/viral-retain dirs
#   bash phase2/run.sh prepare_hiyata_lora # derive Hiyata Host Tropism train/val/test LoRA manifest
#   bash phase2/run.sh kmer_hiyata         # k-mer baseline on the derived Hiyata LoRA manifest
#   bash phase2/run.sh gd                  # classic CE gradient difference, all four conditions
#   bash phase2/run.sh probe_repr          # probe-guided representation training, all four conditions
#   bash phase2/run.sh rmu                 # all three RMU conditions
#   bash phase2/run.sh verify_retain       # verify canonical retain.csv still contains GUE + viral retain rows
#   bash phase2/run.sh probe_nullspace     # projection-only localized probe baseline
#   bash phase2/run.sh projection_screen   # projection + internal eval + pilot HVUE/GUE benchmark screen
#   bash phase2/run.sh probe_guided        # probe-guided localized training (defaults to projection init)
#   bash phase2/run.sh rmu_primary         # localized-primary nonhuman RMU sweep with explicit loss layers 5-9
#   bash phase2/run.sh eval                # internal diagnostic eval for every checkpoint
#   bash phase2/run.sh benchmarks          # primary HVUE LoRA benchmark eval for base + checkpoints
#   bash phase2/run.sh benchmark_pilot     # stratified LoRA pilot eval for explicit/all candidate checkpoints
#   bash phase2/run.sh benchmark_full_top  # full eval for top pilot-ranked checkpoints
#   bash phase2/run.sh taxonomy_heldout    # taxonomy-held-out eval for base + tuned checkpoints
#   bash phase2/run.sh all                 # everything in sequence

set -euo pipefail

PHASE2_PYTHON=${PHASE2_PYTHON:-${PROJECT_PYTHON:-python}}
CKPT_ROOT=data/phase2/checkpoints
BENCH_CKPT_ROOT=${BENCH_CKPT_ROOT:-data/phase2/checkpoints_tuned}
STEPS=${STEPS:-1000}
LR=${LR:-1e-5}
BATCH=${BATCH:-2}
MAX_LEN=${MAX_LEN:-512}
SAVE_STEPS=${SAVE_STEPS:-}
TARGET_MANIFEST=${TARGET_MANIFEST:-data/host_tropism/manifest.csv}
EXTRA_FORGET_MANIFEST=${EXTRA_FORGET_MANIFEST:-data/family_targets/coronaviridae/manifest.csv}
LOCALIZED_LAYERS_PATH=${LOCALIZED_LAYERS_PATH:-data/family_targets/coronaviridae/localized_layers.json}
INTERNAL_TARGET_CONFIG=${INTERNAL_TARGET_CONFIG:-phase2/internal_eval_targets.json}
SPLIT_DIR=${SPLIT_DIR:-data/phase2/splits}
BENCHMARK_MANIFEST=${BENCHMARK_MANIFEST:-data/benchmarks/hvue_gue_manifest.csv}
VIRAL_RETAIN_ROOT=${VIRAL_RETAIN_ROOT:-}
VIRAL_RETAIN_TASKS=${VIRAL_RETAIN_TASKS:-virus_vs_nonvirus dna_vs_rna_virus host_range_prediction hiv1_vs_hiv2 sars_cov_2_lineage_typing influenza_subtype_typing}
VIROBENCH_TASKS=${VIROBENCH_TASKS:-virobench_all_taxon_genus virobench_all_taxon_times virobench_dna_taxon_genus virobench_dna_taxon_times virobench_rna_taxon_genus virobench_rna_taxon_times}
BENCH_BATCH=${BENCH_BATCH:-1}
BENCH_CPU_THREADS=${BENCH_CPU_THREADS:-16}
BENCH_AUTO_BATCH=${BENCH_AUTO_BATCH:-96}
BENCH_PROBE_JOBS=${BENCH_PROBE_JOBS:-7}
BENCH_LAYERS=${BENCH_LAYERS:-5-9}
BENCH_EPOCHS=${BENCH_EPOCHS:-3}
BENCH_MAX_STEPS=${BENCH_MAX_STEPS:-0}
BENCH_EVAL_EVERY=${BENCH_EVAL_EVERY:-100}
BENCH_PATIENCE=${BENCH_PATIENCE:-3}
BENCH_LR=${BENCH_LR:-1e-4}
BENCH_WEIGHT_DECAY=${BENCH_WEIGHT_DECAY:-0.0}
BENCH_LORA_RANK=${BENCH_LORA_RANK:-8}
BENCH_LORA_ALPHA=${BENCH_LORA_ALPHA:-16}
BENCH_LORA_DROPOUT=${BENCH_LORA_DROPOUT:-0.0}
BENCH_METRIC_FOR_BEST=${BENCH_METRIC_FOR_BEST:-auto}
BENCH_PROGRESS_EVERY=${BENCH_PROGRESS_EVERY:-1}
BENCHMARK_SCOPE=${BENCHMARK_SCOPE:-all}
BENCH_TASK_FILTER=${BENCH_TASK_FILTER:-}
BENCH_PILOT_ROOT=${BENCH_PILOT_ROOT:-data/phase2/benchmark_pilot_lora}
BENCH_FULL_OUT_ROOT=${BENCH_FULL_OUT_ROOT:-data/phase2/full_benchmarks_lora}
BENCH_PILOT_MANIFEST=${BENCH_PILOT_MANIFEST:-data/benchmarks/hvue_gue_pilot_manifest.csv}
BENCH_PILOT_CANDIDATES=${BENCH_PILOT_CANDIDATES:-}
BENCH_PILOT_DISCOVER_CANDIDATES=${BENCH_PILOT_DISCOVER_CANDIDATES:-0}
BENCH_PILOT_TOP_K=${BENCH_PILOT_TOP_K:-2}
BENCH_PILOT_TRAIN_PER_LABEL=${BENCH_PILOT_TRAIN_PER_LABEL:-2000}
BENCH_PILOT_VAL_PER_LABEL=${BENCH_PILOT_VAL_PER_LABEL:-500}
BENCH_PILOT_TEST_PER_LABEL=${BENCH_PILOT_TEST_PER_LABEL:-1500}
HIYATA_SOURCE_MANIFEST=${HIYATA_SOURCE_MANIFEST:-data/host_tropism_hiyata/manifest_no_gemini.csv}
HIYATA_LORA_MANIFEST=${HIYATA_LORA_MANIFEST:-data/host_tropism_hiyata/eval_manifest_lora.csv}
HIYATA_LORA_AUDIT=${HIYATA_LORA_AUDIT:-data/host_tropism_hiyata/eval_manifest_lora_audit.json}
HIYATA_HVUE_MANIFEST=${HIYATA_HVUE_MANIFEST:-data/benchmarks/final_fast_eval_manifest.csv}
HIYATA_VAL_FRACTION=${HIYATA_VAL_FRACTION:-0.15}
HIYATA_SEED=${HIYATA_SEED:-42}
KMER_OUT=${KMER_OUT:-data/phase2/kmer_baselines/hiyata_host_tropism_kmer_1-4_binary.csv}
KMER_MIN=${KMER_MIN:-1}
KMER_MAX=${KMER_MAX:-4}
KMER_C_GRID=${KMER_C_GRID:-0.1,1,10}
KMER_MAX_ITER=${KMER_MAX_ITER:-1000}
DOWNLOAD_HVUE=${DOWNLOAD_HVUE:-1}
DOWNLOAD_GUE=${DOWNLOAD_GUE:-0}
DOWNLOAD_VIROBENCH=${DOWNLOAD_VIROBENCH:-0}
TAXONOMY_DATASET=${TAXONOMY_DATASET:-host_tropism}
TAXONOMY_GROUP_KEY=${TAXONOMY_GROUP_KEY:-auto}
TAXONOMY_CKPT_ROOT=${TAXONOMY_CKPT_ROOT:-data/phase2/checkpoints_tuned}
TAXONOMY_OUT_ROOT=${TAXONOMY_OUT_ROOT:-data/phase2/taxonomy_heldout}
TAXONOMY_CINI_INPUT=${TAXONOMY_CINI_INPUT:-$BENCHMARK_MANIFEST}
TAXONOMY_MANIFEST=${TAXONOMY_MANIFEST:-data/host_tropism/manifest.csv}
RMU_CONDITIONS=${RMU_CONDITIONS:-localized full}
RMU_TARGET_DIRECTION=${RMU_TARGET_DIRECTION:-nonhuman}
RMU_DIRECTION_SEQS=${RMU_DIRECTION_SEQS:-500}
PROBE_NULLSPACE_RUN=${PROBE_NULLSPACE_RUN:-probe_nullspace_joint_l5_l9}
PROBE_GUIDED_RUN=${PROBE_GUIDED_RUN:-probe_guided_projinit_ar5_s200}
PROBE_INIT_RUN=${PROBE_INIT_RUN:-$PROBE_NULLSPACE_RUN}
GD_INIT_RUN=${GD_INIT_RUN:-}
GD_ALPHA_FORGET=${GD_ALPHA_FORGET:-1.0}
GD_ALPHA_RETAIN=${GD_ALPHA_RETAIN:-5.0}
GD_FORGET_LOSS_CAP=${GD_FORGET_LOSS_CAP:-0.0}
PROBE_REPR_INIT_RUN=${PROBE_REPR_INIT_RUN:-$PROBE_NULLSPACE_RUN}
RMU_PRIMARY_CONFIG=${RMU_PRIMARY_CONFIG:-phase2/sweep_configs/rmu_localized_nonhuman.json}

verify_retain() {
    "$PHASE2_PYTHON" -u phase2/verify_retain_set.py \
        --csv "$SPLIT_DIR/retain.csv" \
        --summary-json "$SPLIT_DIR/retain_audit.json"
}

ensure_pilot_manifest() {
    if [ -f "$BENCH_PILOT_MANIFEST" ]; then
        echo "[phase2] using existing pilot manifest $BENCH_PILOT_MANIFEST"
        return
    fi
    "$PHASE2_PYTHON" -u phase2/subsample_benchmark_manifest.py \
        --input-manifest "$BENCHMARK_MANIFEST" \
        --output-manifest "$BENCH_PILOT_MANIFEST" \
        --seed "$HIYATA_SEED" \
        --train-per-label "$BENCH_PILOT_TRAIN_PER_LABEL" \
        --val-per-label "$BENCH_PILOT_VAL_PER_LABEL" \
        --test-per-label "$BENCH_PILOT_TEST_PER_LABEL"
}

# Classic gradient difference: -alpha_forget * CE(forget) + alpha_retain * CE(retain).
# This is the objective that produced the archived lora_gd_* results. It takes no
# probe target config; set GD_INIT_RUN="" to train from the base model.
run_gd() {
    for cond in localized probe random full; do
        echo "=== GD $cond ==="
        local init_args=()
        if [ -n "$GD_INIT_RUN" ]; then
            init_args=(--init-from-run "$GD_INIT_RUN")
        fi
        "$PHASE2_PYTHON" -u phase2/unlearn_gd.py \
            --forget-csv "$SPLIT_DIR/forget.csv" \
            --retain-csv "$SPLIT_DIR/retain.csv" \
            --out-dir "$CKPT_ROOT" \
            --run-name "gd_$cond" \
            "${init_args[@]}" \
            --condition "$cond" \
            --steps "$STEPS" --lr "$LR" \
            --alpha-forget "$GD_ALPHA_FORGET" --alpha-retain "$GD_ALPHA_RETAIN" \
            --forget-loss-cap "$GD_FORGET_LOSS_CAP" \
            --batch-size "$BATCH" --max-length "$MAX_LEN" \
            --save-steps "$SAVE_STEPS" \
            --localized-layers-path "$LOCALIZED_LAYERS_PATH"
    done
}

# Probe-guided representation training. Formerly the contents of unlearn_gd.py;
# this is the objective behind the archived refseq_gd_projinit_* runs.
run_probe_repr() {
    for cond in localized probe random full; do
        echo "=== probe-repr $cond ==="
        "$PHASE2_PYTHON" -u phase2/unlearn_probe_repr.py \
            --forget-csv "$SPLIT_DIR/forget.csv" \
            --retain-csv "$SPLIT_DIR/retain.csv" \
            --internal-target-config "$INTERNAL_TARGET_CONFIG" \
            --out-dir "$CKPT_ROOT" \
            --run-name "probe_repr_projinit_$cond" \
            --init-from-run "$PROBE_REPR_INIT_RUN" \
            --condition "$cond" \
            --steps "$STEPS" --lr "$LR" \
            --batch-size "$BATCH" --max-length "$MAX_LEN" \
            --save-steps "$SAVE_STEPS" \
            --localized-layers-path "$LOCALIZED_LAYERS_PATH"
    done
}

run_rmu() {
    for cond in $RMU_CONDITIONS; do
        echo "=== RMU $cond ==="
        "$PHASE2_PYTHON" -u phase2/unlearn_rmu.py \
            --forget-csv "$SPLIT_DIR/forget.csv" \
            --retain-csv "$SPLIT_DIR/retain.csv" \
            --condition "$cond" \
            --target-direction "$RMU_TARGET_DIRECTION" \
            --direction-seqs "$RMU_DIRECTION_SEQS" \
            --steps "$STEPS" --lr "$LR" \
            --batch-size "$BATCH" --max-length "$MAX_LEN" \
            --save-steps "$SAVE_STEPS" \
            --localized-layers-path "$LOCALIZED_LAYERS_PATH"
    done
}

run_probe_nullspace() {
    echo "=== Probe null-space projection ==="
    "$PHASE2_PYTHON" -u phase2/project_probe_nullspace.py \
        --internal-target-config "$INTERNAL_TARGET_CONFIG" \
        --forget-csv "$SPLIT_DIR/forget.csv" \
        --retain-csv "$SPLIT_DIR/retain.csv" \
        --out-dir "$CKPT_ROOT" \
        --run-name "$PROBE_NULLSPACE_RUN" \
        --device "${DEVICE:-cuda:0}"
}

run_probe_guided() {
    echo "=== Probe-guided localized training ==="
    "$PHASE2_PYTHON" -u phase2/unlearn_probe.py \
        --internal-target-config "$INTERNAL_TARGET_CONFIG" \
        --forget-csv "$SPLIT_DIR/forget.csv" \
        --retain-csv "$SPLIT_DIR/retain.csv" \
        --out-dir "$CKPT_ROOT" \
        --run-name "$PROBE_GUIDED_RUN" \
        --init-from-run "$PROBE_INIT_RUN" \
        --device "${DEVICE:-cuda:0}" \
        --steps "$STEPS" --lr "$LR" \
        --batch-size "$BATCH" --max-length "$MAX_LEN" \
        --save-steps "$SAVE_STEPS"
}

run_projection_screen() {
    verify_retain
    ensure_pilot_manifest
    "$PHASE2_PYTHON" -u phase2/run_task2_sweeps.py \
        projection \
        --out-dir "$BENCH_CKPT_ROOT" \
        --run-benchmarks \
        --benchmark-manifest "$BENCH_PILOT_MANIFEST" \
        --benchmark-scope all \
        --device "${DEVICE:-cuda:0}"
}

run_rmu_primary() {
    verify_retain
    "$PHASE2_PYTHON" -u phase2/run_task2_sweeps.py \
        --config "$RMU_PRIMARY_CONFIG" \
        --out-dir "$BENCH_CKPT_ROOT" \
        --device "${DEVICE:-cuda:0}"
}

run_eval() {
    for run in "$CKPT_ROOT"/*/; do
        ckpt="$run/weights.safetensors"
        if [ -f "$ckpt" ]; then
            echo "=== eval $run ==="
            "$PHASE2_PYTHON" -u phase2/eval_unlearn.py \
                --ckpt "$ckpt" \
                --internal-target-config "$INTERNAL_TARGET_CONFIG" \
                --forget-csv "$SPLIT_DIR/forget.csv" \
                --retain-csv "$SPLIT_DIR/retain.csv" \
                --localized-layers-path "$LOCALIZED_LAYERS_PATH"
        fi
    done
}

prepare_benchmarks() {
    local extra_args=()
    if [ "$DOWNLOAD_HVUE" = "1" ]; then
        extra_args+=(--download-hvue)
    fi
    if [ "$DOWNLOAD_GUE" = "1" ]; then
        extra_args+=(--download-gue)
    fi
    if [ "$DOWNLOAD_VIROBENCH" = "1" ]; then
        extra_args+=(--download-virobench)
    fi
    if [ -n "$VIRAL_RETAIN_ROOT" ]; then
        extra_args+=(--viral-retain-root "$VIRAL_RETAIN_ROOT")
    fi

    "$PHASE2_PYTHON" -u phase2/prepare_benchmarks.py \
        "${extra_args[@]}" \
        --raw-root data/benchmarks/raw \
        --out-manifest "$BENCHMARK_MANIFEST" \
        --viral-retain-tasks $VIRAL_RETAIN_TASKS \
        --virobench-tasks $VIROBENCH_TASKS
}

run_audit() {
    "$PHASE2_PYTHON" -u phase2/audit_experiment_state.py \
        --raw-root data/benchmarks/raw \
        --manifest "$BENCHMARK_MANIFEST" \
        --out data/phase2/experiment_audit.json
}

prepare_hiyata_lora() {
    "$PHASE2_PYTHON" -u phase2/prepare_hiyata_lora_manifest.py \
        --input-manifest "$HIYATA_SOURCE_MANIFEST" \
        --out-manifest "$HIYATA_LORA_MANIFEST" \
        --audit-json "$HIYATA_LORA_AUDIT" \
        --hvue-manifest "$HIYATA_HVUE_MANIFEST" \
        --val-fraction "$HIYATA_VAL_FRACTION" \
        --seed "$HIYATA_SEED" \
        --max-length "$MAX_LEN"
}

run_kmer_hiyata() {
    "$PHASE2_PYTHON" -u phase2/eval_kmer_baseline.py \
        --benchmark-manifest "$HIYATA_LORA_MANIFEST" \
        --task-filter host_tropism_hiyata \
        --out-csv "$KMER_OUT" \
        --max-length "$MAX_LEN" \
        --kmer-min "$KMER_MIN" \
        --kmer-max "$KMER_MAX" \
        --kmer-binary \
        --c-grid "$KMER_C_GRID" \
        --max-iter "$KMER_MAX_ITER"
}

run_benchmarks() {
    if [ ! -f "$BENCHMARK_MANIFEST" ]; then
        echo "Benchmark manifest not found: $BENCHMARK_MANIFEST" >&2
        echo "Set BENCHMARK_MANIFEST to a CSV with benchmark,task,split,sequence,label columns." >&2
        exit 1
    fi
    echo "=== benchmark base model ==="
    "$PHASE2_PYTHON" -u phase2/eval_benchmarks.py \
        --benchmark-manifest "$BENCHMARK_MANIFEST" \
        --out-dir data/phase2/base_benchmarks \
        --resume \
        --batch-size "$BENCH_BATCH" \
        --cpu-threads "$BENCH_CPU_THREADS" \
        --epochs "$BENCH_EPOCHS" \
        --max-steps "$BENCH_MAX_STEPS" \
        --eval-every "$BENCH_EVAL_EVERY" \
        --patience "$BENCH_PATIENCE" \
        --lr "$BENCH_LR" \
        --weight-decay "$BENCH_WEIGHT_DECAY" \
        --lora-rank "$BENCH_LORA_RANK" \
        --lora-alpha "$BENCH_LORA_ALPHA" \
        --lora-dropout "$BENCH_LORA_DROPOUT" \
        --metric-for-best "$BENCH_METRIC_FOR_BEST" \
        --progress-every "$BENCH_PROGRESS_EVERY" \
        --max-length "$MAX_LEN"

    for run in "$CKPT_ROOT"/*/; do
        ckpt="$run/weights.safetensors"
        if [ -f "$ckpt" ]; then
            echo "=== benchmark $run ==="
            "$PHASE2_PYTHON" -u phase2/eval_benchmarks.py \
                --ckpt "$ckpt" \
                --benchmark-manifest "$BENCHMARK_MANIFEST" \
                --resume \
                --batch-size "$BENCH_BATCH" \
                --cpu-threads "$BENCH_CPU_THREADS" \
                --epochs "$BENCH_EPOCHS" \
                --max-steps "$BENCH_MAX_STEPS" \
                --eval-every "$BENCH_EVAL_EVERY" \
                --patience "$BENCH_PATIENCE" \
                --lr "$BENCH_LR" \
                --weight-decay "$BENCH_WEIGHT_DECAY" \
                --lora-rank "$BENCH_LORA_RANK" \
                --lora-alpha "$BENCH_LORA_ALPHA" \
                --lora-dropout "$BENCH_LORA_DROPOUT" \
                --metric-for-best "$BENCH_METRIC_FOR_BEST" \
                --progress-every "$BENCH_PROGRESS_EVERY" \
                --max-length "$MAX_LEN"
        fi
    done
}

run_benchmark_pilot() {
    local cmd=(
        "$PHASE2_PYTHON" -u phase2/run_benchmark_pilot.py pilot
        --python "$PHASE2_PYTHON" \
        --full-manifest "$BENCHMARK_MANIFEST" \
        --pilot-manifest "$BENCH_PILOT_MANIFEST" \
        --benchmark-scope "$BENCHMARK_SCOPE" \
        --pilot-root "$BENCH_PILOT_ROOT" \
        --ckpt-root "$BENCH_CKPT_ROOT" \
        --top-k "$BENCH_PILOT_TOP_K" \
        --train-per-label "$BENCH_PILOT_TRAIN_PER_LABEL" \
        --val-per-label "$BENCH_PILOT_VAL_PER_LABEL" \
        --test-per-label "$BENCH_PILOT_TEST_PER_LABEL" \
        --device "${DEVICE:-cuda:0}" \
        --batch-size "$BENCH_BATCH" \
        --cpu-threads "$BENCH_CPU_THREADS" \
        --epochs "$BENCH_EPOCHS" \
        --max-steps "$BENCH_MAX_STEPS" \
        --eval-every "$BENCH_EVAL_EVERY" \
        --patience "$BENCH_PATIENCE" \
        --lr "$BENCH_LR" \
        --weight-decay "$BENCH_WEIGHT_DECAY" \
        --lora-rank "$BENCH_LORA_RANK" \
        --lora-alpha "$BENCH_LORA_ALPHA" \
        --lora-dropout "$BENCH_LORA_DROPOUT" \
        --metric-for-best "$BENCH_METRIC_FOR_BEST" \
        --max-length "$MAX_LEN"
    )
    if [[ -n "$BENCH_TASK_FILTER" ]]; then
        cmd+=(--task-filter "$BENCH_TASK_FILTER")
    fi
    if [[ "$BENCH_PILOT_DISCOVER_CANDIDATES" == "1" ]]; then
        cmd+=(--discover-candidates)
    elif [[ -n "$BENCH_PILOT_CANDIDATES" ]]; then
        local candidates=()
        read -r -a candidates <<< "$BENCH_PILOT_CANDIDATES"
        cmd+=(--candidates "${candidates[@]}")
    else
        echo "BENCH_PILOT_CANDIDATES is empty. Old gd_full_ar5/rmu_full_sc200 defaults were probe-selected legacy candidates." >&2
        echo "Set BENCH_PILOT_CANDIDATES='run_a run_b' for explicit evaluation, or BENCH_PILOT_DISCOVER_CANDIDATES=1 for all checkpoints." >&2
        return 2
    fi
    "${cmd[@]}"
}

run_benchmark_full_top() {
    local cmd=(
        "$PHASE2_PYTHON" -u phase2/run_benchmark_pilot.py full-top
        --python "$PHASE2_PYTHON" \
        --full-manifest "$BENCHMARK_MANIFEST" \
        --benchmark-scope "$BENCHMARK_SCOPE" \
        --pilot-root "$BENCH_PILOT_ROOT" \
        --rankings-json "$BENCH_PILOT_ROOT/pilot_rankings.json" \
        --full-out-root "$BENCH_FULL_OUT_ROOT" \
        --ckpt-root "$BENCH_CKPT_ROOT" \
        --top-k "$BENCH_PILOT_TOP_K" \
        --device "${DEVICE:-cuda:0}" \
        --batch-size "$BENCH_BATCH" \
        --cpu-threads "$BENCH_CPU_THREADS" \
        --epochs "$BENCH_EPOCHS" \
        --max-steps "$BENCH_MAX_STEPS" \
        --eval-every "$BENCH_EVAL_EVERY" \
        --patience "$BENCH_PATIENCE" \
        --lr "$BENCH_LR" \
        --weight-decay "$BENCH_WEIGHT_DECAY" \
        --lora-rank "$BENCH_LORA_RANK" \
        --lora-alpha "$BENCH_LORA_ALPHA" \
        --lora-dropout "$BENCH_LORA_DROPOUT" \
        --metric-for-best "$BENCH_METRIC_FOR_BEST" \
        --max-length "$MAX_LEN"
    )
    if [[ -n "$BENCH_TASK_FILTER" ]]; then
        cmd+=(--task-filter "$BENCH_TASK_FILTER")
    fi
    "${cmd[@]}"
}

run_taxonomy_heldout_base() {
    echo "=== taxonomy-held-out base model ($TAXONOMY_DATASET, group_key=$TAXONOMY_GROUP_KEY) ==="
    "$PHASE2_PYTHON" -u phase2/eval_taxonomy_heldout.py \
        --dataset "$TAXONOMY_DATASET" \
        --manifest "$TAXONOMY_MANIFEST" \
        --cini-input "$TAXONOMY_CINI_INPUT" \
        --group-key "$TAXONOMY_GROUP_KEY" \
        --out-dir "$TAXONOMY_OUT_ROOT/base" \
        --layers "$BENCH_LAYERS" \
        --batch-size "$BENCH_BATCH" \
        --auto-batch-size "$BENCH_AUTO_BATCH" \
        --cpu-threads "$BENCH_CPU_THREADS" \
        --probe-jobs "$BENCH_PROBE_JOBS" \
        --progress-every "$BENCH_PROGRESS_EVERY" \
        --max-length "$MAX_LEN"
}

run_taxonomy_heldout_ckpts() {
    for run in "$TAXONOMY_CKPT_ROOT"/*/; do
        ckpt="$run/weights.safetensors"
        if [ -f "$ckpt" ]; then
            run_name="$(basename "$run")"
            echo "=== taxonomy-held-out $run_name ==="
            "$PHASE2_PYTHON" -u phase2/eval_taxonomy_heldout.py \
                --ckpt "$ckpt" \
                --dataset "$TAXONOMY_DATASET" \
                --manifest "$TAXONOMY_MANIFEST" \
                --cini-input "$TAXONOMY_CINI_INPUT" \
                --group-key "$TAXONOMY_GROUP_KEY" \
                --out-dir "$TAXONOMY_OUT_ROOT/$run_name" \
                --layers "$BENCH_LAYERS" \
                --batch-size "$BENCH_BATCH" \
                --auto-batch-size "$BENCH_AUTO_BATCH" \
                --cpu-threads "$BENCH_CPU_THREADS" \
                --probe-jobs "$BENCH_PROBE_JOBS" \
                --progress-every "$BENCH_PROGRESS_EVERY" \
                --max-length "$MAX_LEN"
        fi
    done
}

run_taxonomy_heldout() {
    run_taxonomy_heldout_base
    run_taxonomy_heldout_ckpts
}

case "${1:-all}" in
    splits) "$PHASE2_PYTHON" -u phase2/build_unlearn_splits.py --manifest "$TARGET_MANIFEST" --extra-forget-manifest "$EXTRA_FORGET_MANIFEST" --out-dir "$SPLIT_DIR" ;;
    audit) run_audit ;;
    prepare_benchmarks) prepare_benchmarks ;;
    prepare_hiyata_lora) prepare_hiyata_lora ;;
    kmer_hiyata) run_kmer_hiyata ;;
    gd) run_gd ;;
    probe_repr) run_probe_repr ;;
    rmu) run_rmu ;;
    verify_retain) verify_retain ;;
    probe_nullspace) run_probe_nullspace ;;
    projection_screen) run_projection_screen ;;
    probe_guided) run_probe_guided ;;
    rmu_primary) run_rmu_primary ;;
    eval) run_eval ;;
    benchmarks) run_benchmarks ;;
    benchmark_pilot) run_benchmark_pilot ;;
    benchmark_full_top) run_benchmark_full_top ;;
    taxonomy_heldout_base) run_taxonomy_heldout_base ;;
    taxonomy_heldout_ckpts) run_taxonomy_heldout_ckpts ;;
    taxonomy_heldout) run_taxonomy_heldout ;;
    all)
        "$PHASE2_PYTHON" -u phase2/build_unlearn_splits.py --manifest "$TARGET_MANIFEST" --extra-forget-manifest "$EXTRA_FORGET_MANIFEST" --out-dir "$SPLIT_DIR"
        run_audit
        prepare_benchmarks
        prepare_hiyata_lora
        run_kmer_hiyata
        run_probe_nullspace
        run_probe_guided
        run_gd
        run_probe_repr
        run_rmu
        run_eval
        run_benchmarks
        ;;
    *)
        echo "Unknown target: $1"
        echo "Usage: bash phase2/run.sh [splits|audit|prepare_benchmarks|prepare_hiyata_lora|kmer_hiyata|gd|probe_repr|rmu|verify_retain|probe_nullspace|projection_screen|probe_guided|rmu_primary|eval|benchmarks|benchmark_pilot|benchmark_full_top|taxonomy_heldout_base|taxonomy_heldout_ckpts|taxonomy_heldout|all]"
        exit 1
        ;;
esac
