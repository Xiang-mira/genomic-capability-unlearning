# python phase1/download_refseq.py \
#   --out-dir data/phase1 \
#   --target-per-class 10000 \
#   --max-length 2048 \
#   --viral-max-files 4 \
#   --nonviral-max-files 4 \
#   --nonviral-group bacteria \
#   --manifest-only


# # 2) Extract features
# python phase1/extract_features.py \
#   --manifest data/phase1/manifest.csv \
#   --model-dir ./evo-1-8k-base \
#   --config-path configs/evo-1-8k-base_inference.yml \
#   --out-dir data/phase1/features \
#   --batch-size 80 \
#   --max-length 512

# # 3) Train probes and write CSV
# python phase1/train_probes.py \
#   --feature-dir data/phase1/features \
#   --out-dir data/phase1/probes \
#   --c-grid 0.1,1,10

# # 4) Plot
# python phase1/plot_metrics.py \
#   --metrics data/phase1/probes/probe_metrics_by_layer.csv \
#   --out-dir data/phase1/probes




python phase1/build_host_tropism_dataset.py \
    --target-per-class 5000 \
    --max-length 512 \
    --manifest-only \
    --max-per-virus-tax-id 50

# #baseline
# python phase1/baseline_gc_1gram.py --manifest data/host_tropism/manifest.csv --out-dir data/host_tropism/baselines --feature gc_1gram_length


# python phase1/baseline_gc_1gram.py --manifest data/host_tropism/manifest.csv --out-dir data/host_tropism/baselines --feature kmer --kmer-max 4 --kmer-binary --max-iter 1000

  # 然后跑 probe：

  python phase1/extract_features.py \
    --batch-size 80 \
    --max-length 512 \
    --representation next_norm

  python phase1/diagnose_features.py

  python phase1/train_probes.py \
    --c-grid 0.001,0.01,0.1,1 \
    --max-iter 1000

  python phase1/plot_metrics.py