# Causal-Chain Validation Plan for Phase 2

## Primary Claim

This phase is designed to validate the causal chain:

`Host-tropism knowledge -> model representation -> probe measurement -> GD/RMU unlearning -> downstream host-tropism behavior -> general viral capability retention`.

The goal is not simply to fill benchmark tables. Every artifact should support one of the claims below.

## Objective 1: Host-Tropism Target Validity

Scientific claim: host-tropism is real, model-internal information that can be decoded from Evo representations and is not only a dataset shortcut.

Primary target-validity dataset: `hiyata/Virus-Host-Genomes`, converted locally with `phase2/prepare_hiyata_host_tropism.py`. This dataset is preferred over the legacy local manifest because it includes `family` and `genus`, enabling real family/genus-controlled splits.

Dataset roles:

- `hiyata/Virus-Host-Genomes`: primary controlled-split target-validity evidence.
- `data/host_tropism/manifest.csv`: legacy continuity with Phase 1 layer localization and previous Phase 2 diagnostics.
- `duttaprat/HVUE` Host Tropism: external benchmark/downstream validation, not the only target-validity dataset.

Run the Base model first on controlled host-tropism splits:

| Split | Confound tested | Evidence produced |
| --- | --- | --- |
| `random` | No confound removed; baseline learnability | Whether host-tropism is decodable at all |
| `taxonomy` | Train/test overlap at selected taxonomy key | Whether decoding survives held-out virus/taxon groups |
| `homology` | Near-duplicate or sequence-similarity leakage | Whether decoding survives approximate sequence-cluster holdout |
| `within_group` | Between-group taxonomy shortcut | Whether human/non-human remains predictable inside mixed-label groups |

Important limitation for the legacy local manifest: it contains `virus_tax_id`, `virus_name`, and `source`, but not a full lineage table. A split with `--group-key virus_tax_id` is taxon/species-like held-out evidence, not a true family-held-out evaluation. Hiyata should be used for the primary family/genus-held-out evidence.

Success criterion: host-tropism remains meaningfully predictable under at least one controlled split, not only random split.

## Objective 2: Probe Metric Validity

Scientific claim: frozen linear probe scores are useful only if they predict downstream supervised fine-tuning behavior.

Run probe-vs-SFT with identical train/val/test rows for:

- `hvue_human_host_tropism`
- one controlled host-tropism split exported by `eval_taxonomy_heldout.py`
- `hvue_human_virus_pathogenicity_cini`
- `gue_prom_300_all`
- `virobench_all_taxon_genus`

Models:

- Base
- `gd_localized_ar5_s1000`
- `gd_random_ar5`
- `rmu_localized_sc50_l4`
- `rmu_random_sc50`

Report:

- frozen probe score
- SFT best validation step
- SFT test mean/std across seeds `42,43,44`
- Pearson correlation
- Spearman/rank consistency
- correlation between probe degradation from Base and SFT degradation from Base

Success criterion: probe changes correlate with downstream SFT behavior, especially for host-tropism.

## Objective 3: GD vs RMU Trade-Off

Scientific claim: GD and RMU should be compared by host-tropism removal versus collateral damage.

Always report together:

- host-tropism forgetting
- controlled host-tropism downstream degradation
- retain validation loss/PPL
- GUE retain score
- ViroBench viral-retain score

Do not interpret lower probe AUROC alone as successful unlearning.

## Objective 4: Mechanistic Trajectories

Scientific claim: trajectories should reveal when forgetting, probe collapse, downstream degradation, and retain damage occur.

Use checkpoints at steps `100,200,500,1000` for:

- `gd_localized_ar5_s1000`
- `gd_random_ar5`
- `rmu_localized_sc50_l4`
- `rmu_random_sc50`

Prioritized stepwise outputs:

- internal fixed-probe AUROC
- forget/retain validation loss and PPL
- host-tropism probe/SFT subset
- selected GUE/ViroBench retain tasks
- full HVUE/GUE/ViroBench only after target/probe validity is established

Interpret:

- whether probe collapse precedes SFT degradation
- whether forgetting precedes retain damage
- whether GD and RMU follow different dynamics

## Execution Priority

1. Base host-tropism controlled split validation.
2. Probe-vs-SFT on Base plus final GD/RMU localized/random checkpoints.
3. GD/RMU trade-off table with forgetting and retention side by side.
4. Stepwise trajectory completeness.

ViroBench is retained as evidence for general viral capability preservation. It is not evidence that host-tropism is a valid unlearning target.
