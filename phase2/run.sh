#!/usr/bin/env bash
# Phase 2 orchestration: splits -> unlearning runs -> per-run evaluation.
# Usage:
#   bash phase2/run.sh splits              # one-time: build forget/retain CSVs
#   bash phase2/run.sh audit               # dataset / manifest / checkpoint availability audit
#   bash phase2/run.sh prepare_benchmarks  # rebuild benchmark manifest from raw HVUE/GUE/viral-retain dirs
#   bash phase2/run.sh gd                  # all four GD conditions (full, localized, probe, random)
#   bash phase2/run.sh rmu                 # all three RMU conditions
#   bash phase2/run.sh eval                # internal diagnostic eval for every checkpoint
#   bash phase2/run.sh benchmarks          # external HVUE/GUE/viral-retain benchmark eval for base + checkpoints
#   bash phase2/run.sh benchmark_pilot     # stratified pilot eval for selected candidate checkpoints
#   bash phase2/run.sh benchmark_full_top  # full eval for top pilot-ranked checkpoints
#   bash phase2/run.sh taxonomy_heldout    # taxonomy-held-out eval for base + tuned checkpoints
#   bash phase2/run.sh all                 # everything in sequence

set -euo pipefail

PHASE2_PYTHON=${PHASE2_PYTHON:-python}
CKPT_ROOT=data/phase2/checkpoints
STEPS=${STEPS:-200}
LR=${LR:-1e-5}
BATCH=${BATCH:-2}
MAX_LEN=${MAX_LEN:-512}
BENCHMARK_MANIFEST=${BENCHMARK_MANIFEST:-data/benchmarks/hvue_gue_manifest.csv}
VIRAL_RETAIN_ROOT=${VIRAL_RETAIN_ROOT:-data/benchmarks/raw/viral_retain}
VIRAL_RETAIN_TASKS=${VIRAL_RETAIN_TASKS:-host_range_prediction dna_vs_rna_virus hiv1_vs_hiv2}
BENCH_BATCH=${BENCH_BATCH:-0}
BENCH_AUTO_BATCH=${BENCH_AUTO_BATCH:-96}
BENCH_CPU_THREADS=${BENCH_CPU_THREADS:-16}
BENCH_PROBE_JOBS=${BENCH_PROBE_JOBS:-7}
BENCH_LAYERS=${BENCH_LAYERS:-3-9}
BENCH_PROGRESS_EVERY=${BENCH_PROGRESS_EVERY:-25000}
BENCH_FEATURE_CACHE_DIR=${BENCH_FEATURE_CACHE_DIR:-}
BENCH_PILOT_ROOT=${BENCH_PILOT_ROOT:-data/phase2/benchmark_pilot}
BENCH_PILOT_MANIFEST=${BENCH_PILOT_MANIFEST:-data/benchmarks/hvue_gue_pilot_manifest.csv}
BENCH_PILOT_CANDIDATES=${BENCH_PILOT_CANDIDATES:-gd_full_ar5 gd_localized_ar5_s1000 rmu_full_sc200 rmu_full_sc100}
BENCH_PILOT_TOP_K=${BENCH_PILOT_TOP_K:-2}
BENCH_PILOT_TRAIN_PER_LABEL=${BENCH_PILOT_TRAIN_PER_LABEL:-2000}
BENCH_PILOT_VAL_PER_LABEL=${BENCH_PILOT_VAL_PER_LABEL:-500}
BENCH_PILOT_TEST_PER_LABEL=${BENCH_PILOT_TEST_PER_LABEL:-1500}
DOWNLOAD_HVUE=${DOWNLOAD_HVUE:-1}
DOWNLOAD_GUE=${DOWNLOAD_GUE:-0}
TAXONOMY_DATASET=${TAXONOMY_DATASET:-host_tropism}
TAXONOMY_GROUP_KEY=${TAXONOMY_GROUP_KEY:-auto}
TAXONOMY_CKPT_ROOT=${TAXONOMY_CKPT_ROOT:-data/phase2/checkpoints_tuned}
TAXONOMY_OUT_ROOT=${TAXONOMY_OUT_ROOT:-data/phase2/taxonomy_heldout}
TAXONOMY_CINI_INPUT=${TAXONOMY_CINI_INPUT:-$BENCHMARK_MANIFEST}
TAXONOMY_MANIFEST=${TAXONOMY_MANIFEST:-data/host_tropism/manifest.csv}

run_gd() {
    for cond in localized probe random full; do
        echo "=== GD $cond ==="
        python phase2/unlearn_gd.py \
            --condition "$cond" \
            --steps "$STEPS" --lr "$LR" \
            --batch-size "$BATCH" --max-length "$MAX_LEN"
    done
}

run_rmu() {
    for cond in localized random full; do
        echo "=== RMU $cond ==="
        python phase2/unlearn_rmu.py \
            --condition "$cond" \
            --steps "$STEPS" --lr "$LR" \
            --batch-size "$BATCH" --max-length "$MAX_LEN"
    done
}

run_eval() {
    for run in "$CKPT_ROOT"/*/; do
        ckpt="$run/weights.safetensors"
        if [ -f "$ckpt" ]; then
            echo "=== eval $run ==="
            python phase2/eval_unlearn.py --ckpt "$ckpt"
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

    python phase2/prepare_benchmarks.py \
        "${extra_args[@]}" \
        --raw-root data/benchmarks/raw \
        --out-manifest "$BENCHMARK_MANIFEST" \
        --viral-retain-root "$VIRAL_RETAIN_ROOT" \
        --viral-retain-tasks $VIRAL_RETAIN_TASKS
}

run_audit() {
    python phase2/audit_experiment_state.py \
        --raw-root data/benchmarks/raw \
        --manifest "$BENCHMARK_MANIFEST" \
        --out data/phase2/experiment_audit.json
}

run_benchmarks() {
    if [ ! -f "$BENCHMARK_MANIFEST" ]; then
        echo "Benchmark manifest not found: $BENCHMARK_MANIFEST" >&2
        echo "Set BENCHMARK_MANIFEST to a CSV with benchmark,task,split,sequence,label columns." >&2
        exit 1
    fi
    local cache_args=()
    if [ -n "$BENCH_FEATURE_CACHE_DIR" ]; then
        cache_args+=(--feature-cache-dir "$BENCH_FEATURE_CACHE_DIR")
    fi

    echo "=== benchmark base model ==="
    python -u phase2/eval_benchmarks.py \
        --benchmark-manifest "$BENCHMARK_MANIFEST" \
        --out-dir data/phase2/base_benchmarks \
        --resume \
        --layers "$BENCH_LAYERS" \
        --batch-size "$BENCH_BATCH" \
        --auto-batch-size "$BENCH_AUTO_BATCH" \
        --cpu-threads "$BENCH_CPU_THREADS" \
        --probe-jobs "$BENCH_PROBE_JOBS" \
        --progress-every "$BENCH_PROGRESS_EVERY" \
        --max-length "$MAX_LEN" \
        "${cache_args[@]}"

    for run in "$CKPT_ROOT"/*/; do
        ckpt="$run/weights.safetensors"
        if [ -f "$ckpt" ]; then
            echo "=== benchmark $run ==="
            python -u phase2/eval_benchmarks.py \
                --ckpt "$ckpt" \
                --benchmark-manifest "$BENCHMARK_MANIFEST" \
                --resume \
                --layers "$BENCH_LAYERS" \
                --batch-size "$BENCH_BATCH" \
                --auto-batch-size "$BENCH_AUTO_BATCH" \
                --cpu-threads "$BENCH_CPU_THREADS" \
                --probe-jobs "$BENCH_PROBE_JOBS" \
                --progress-every "$BENCH_PROGRESS_EVERY" \
                --max-length "$MAX_LEN" \
                "${cache_args[@]}"
        fi
    done
}

run_benchmark_pilot() {
    "$PHASE2_PYTHON" -u phase2/run_benchmark_pilot.py pilot \
        --python "$PHASE2_PYTHON" \
        --full-manifest "$BENCHMARK_MANIFEST" \
        --pilot-manifest "$BENCH_PILOT_MANIFEST" \
        --pilot-root "$BENCH_PILOT_ROOT" \
        --ckpt-root data/phase2/checkpoints_tuned \
        --top-k "$BENCH_PILOT_TOP_K" \
        --train-per-label "$BENCH_PILOT_TRAIN_PER_LABEL" \
        --val-per-label "$BENCH_PILOT_VAL_PER_LABEL" \
        --test-per-label "$BENCH_PILOT_TEST_PER_LABEL" \
        --device "${DEVICE:-cuda:0}" \
        --layers "$BENCH_LAYERS" \
        --batch-size "$BENCH_BATCH" \
        --auto-batch-size "$BENCH_AUTO_BATCH" \
        --cpu-threads "$BENCH_CPU_THREADS" \
        --probe-jobs "$BENCH_PROBE_JOBS" \
        --progress-every "$BENCH_PROGRESS_EVERY" \
        --max-length "$MAX_LEN" \
        --candidates $BENCH_PILOT_CANDIDATES
}

run_benchmark_full_top() {
    local extra_args=()
    if [ -n "$BENCH_FEATURE_CACHE_DIR" ]; then
        extra_args+=(--full-feature-cache-dir "$BENCH_FEATURE_CACHE_DIR")
    fi
    "$PHASE2_PYTHON" -u phase2/run_benchmark_pilot.py full-top \
        --python "$PHASE2_PYTHON" \
        --full-manifest "$BENCHMARK_MANIFEST" \
        --pilot-root "$BENCH_PILOT_ROOT" \
        --rankings-json "$BENCH_PILOT_ROOT/pilot_rankings.json" \
        --ckpt-root data/phase2/checkpoints_tuned \
        --top-k "$BENCH_PILOT_TOP_K" \
        --device "${DEVICE:-cuda:0}" \
        --layers "$BENCH_LAYERS" \
        --batch-size "$BENCH_BATCH" \
        --auto-batch-size "$BENCH_AUTO_BATCH" \
        --cpu-threads "$BENCH_CPU_THREADS" \
        --probe-jobs "$BENCH_PROBE_JOBS" \
        --progress-every "$BENCH_PROGRESS_EVERY" \
        --max-length "$MAX_LEN" \
        "${extra_args[@]}"
}

run_taxonomy_heldout_base() {
    echo "=== taxonomy-held-out base model ($TAXONOMY_DATASET, group_key=$TAXONOMY_GROUP_KEY) ==="
    python -u phase2/eval_taxonomy_heldout.py \
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
            python -u phase2/eval_taxonomy_heldout.py \
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
    splits) python phase2/build_unlearn_splits.py ;;
    audit) run_audit ;;
    prepare_benchmarks) prepare_benchmarks ;;
    gd) run_gd ;;
    rmu) run_rmu ;;
    eval) run_eval ;;
    benchmarks) run_benchmarks ;;
    benchmark_pilot) run_benchmark_pilot ;;
    benchmark_full_top) run_benchmark_full_top ;;
    taxonomy_heldout_base) run_taxonomy_heldout_base ;;
    taxonomy_heldout_ckpts) run_taxonomy_heldout_ckpts ;;
    taxonomy_heldout) run_taxonomy_heldout ;;
    all)
        python phase2/build_unlearn_splits.py
        run_audit
        prepare_benchmarks
        run_gd
        run_rmu
        run_eval
        run_benchmarks
        ;;
    *)
        echo "Unknown target: $1"
        echo "Usage: bash phase2/run.sh [splits|audit|prepare_benchmarks|gd|rmu|eval|benchmarks|benchmark_pilot|benchmark_full_top|taxonomy_heldout_base|taxonomy_heldout_ckpts|taxonomy_heldout|all]"
        exit 1
        ;;
esac
