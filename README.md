# ORDNA training on strict_5km_3fold

Self-supervised ORDNA triplet encoder trained independently within each spatial
fold. This repository prepares embeddings; it does not train the protection
classifier. It is an adaptation for classification, not a claim of reproducing
all paper settings. See [PROVENANCE.md](PROVENANCE.md).

## Splits

| Fold | Train | Validation (frozen inference only) |
|---|---:|---:|
|01|290|142|
|02|286|146|
|03|288|144|

Only `train` IDs can be anchors, positives or negatives. The training loop never
reads validation sequences. Export later embeds both roles using the frozen model.
Each raw file must be named `<spygen_code>.csv` with columns `Forward,Reverse`.
Files may be nested; duplicate IDs are rejected. Unlisted raw samples are ignored.
No metadata/target CSV or species matrix is needed.

## Install on the server

From the repository root:

```bash
conda env create -f environment.yml
conda activate ordna-training
python -m unittest discover -s tests -v
```

Alternatively install into a suitable existing Python3.10 environment with
`pip install -e .`. The environment file pins the versions tested locally. For a
server-specific CUDA build, provision its compatible PyTorch environment first,
then install this package. No CUDA toolkit or GPU is installed by these scripts.

## Check paths before training

```bash
ordna-train audit --samples-dir /server/raw_reads \
  --split splits/strict_5km_3fold/index_mpa_merged_3_fold_01.csv
```

`audit` checks IDs/paths only. Train performs full raw-read QC on training samples.
Reads are normalized to lowercase; empty/non-IUPAC pairs are discarded and counted.
No truncation: a valid read longer than `--sequence-length` causes an error. Choose
this length from the sequencing/preprocessing protocol, not validation labels.
At least2000 valid pairs per training sample are required for the default1000-read
anchor/positive subsets. Insufficient training samples are excluded automatically and recorded in
`training/excluded/excluded_samples.csv`, including samples with zero valid pairs.
Validation roles are not changed. Missing files and reads exceeding the configured
length still cause errors. Excluded training IDs are also omitted from embedding
export, so they cannot later be mistaken for trained-on samples. The two samples previously excluded from BT must be checked again
because these splits contain432 rather than430 IDs.

## GPU pilot, then fixed-budget runs

Replace server paths and sequence length below. The illustrative length150 must
be checked against the actual reads. First test one step, batch1, in a dedicated
output directory; this is only a hardware/IO pilot, not a scientific result:

```bash
ordna-train train --samples-dir /server/raw_reads \
  --split splits/strict_5km_3fold/index_mpa_merged_3_fold_01.csv \
  --cache-dir /server/ordna_cache --output /server/results/pilot \
  --sequence-length 150 --steps-per-epoch 1 --epochs 1 --batch-size 1 --device cuda
```

After deciding an update budget and batch size, train/export one fold at a time.
The example200 steps x20 epochs is an explicit exploratory budget, **not a tuned
recommendation or an equivalent to the full upstream epoch budget**:

```bash
bash scripts/run_fold.sh 01 /server/raw_reads /server/results/ordna_seed0 /server/ordna_cache 150 200 20 --batch-size 8 --device cuda --seed 0
bash scripts/run_fold.sh 02 /server/raw_reads /server/results/ordna_seed0 /server/ordna_cache 150 200 20 --batch-size 8 --device cuda --seed 0
bash scripts/run_fold.sh 03 /server/raw_reads /server/results/ordna_seed0 /server/ordna_cache 150 200 20 --batch-size 8 --device cuda --seed 0
```

Training uses the last epoch of the declared budget. No early stopping or model
selection uses protection labels or validation features. Repeating seed1/2 needs
different output roots. Output directories must be fresh; no automatic resume or
overwrite is performed. An interrupted training retains its last completed epoch
checkpoint in `training/checkpoints/last.pt`; export that checkpoint explicitly if desired.

The token cache is a deterministic disk-backed preprocessing cache, never learned
across samples. It can be shared between runs. Expect disk usage around
`valid_pairs * 2 * sequence_length` bytes per sample, plus overhead. Cache keys use
file path/size/mtime, token map and sequence length; QC also records a normalized
valid-pair content hash. Training does not eagerly copy all reads into RAM.

## Export

```bash
ordna-train export --samples-dir /server/raw_reads \
  --split splits/strict_5km_3fold/index_mpa_merged_3_fold_01.csv \
  --checkpoint /server/results/ordna_seed0/fold_01/training/checkpoints/last.pt \
  --output /server/results/ordna_seed0/fold_01/embeddings \
  --cache-dir /server/ordna_cache --device cuda
```

Default: export all valid reads once in shuffled non-overlapping subsets, including
the last partial subset. `--max-subsets N` limits export toN subsets per sample;
do not mix export policies when comparing models. This option does not cap training.
The shell wrapper exports all reads; a different policy requires explicit export.

Files: `embeddings_projector_64D.csv`, `embeddings_preprojector_256D.csv`,
`embedding_qc.csv`, `export_config.json`. Feature CSVs contain `Sample,role,Dim1...`
and `Standard_dev_1...`; role names are `train` and `outer_eval` for compatibility
with the existing classifier. They contain no targets. Full embeddings have one row
per supplied split ID, including validation. Models trained on different folds have
different coordinate systems: never concatenate their coordinates to train one
classifier. Fit one classifier per encoder/fold, aggregate held-out predictions.

## Comparing to Barlow Twins

Train BT again on these exact new train IDs before claiming a paired method
comparison. Existing323/107 results use different splits and are not controls for
these290/142,286/146,288/144 folds. Match input reads, subset size, architecture,
export/partial-subset policy, seeds and a documented compute budget. BT needs two
views and triplet training needs three, so equal step counts are not equal FLOPs.
Freeze encoders before downstream classifiers; fit classifier scalers/loss weights
on training only. The habitat-holdout samples remain outside the supplied splits.

This package has no W&B dependency: config/QC/history/checkpoints are saved locally
on the server for audit. No private reads, paper PDF, caches or checkpoints are
included in this repository. Only publish sample IDs if project permissions allow.

## Run folder and exclusions

Each fold has this layout:

```text
fold_01/
  training/
    config.json
    history.json
    training_qc.csv
    sample_roles_original.csv
    sample_roles.csv
    excluded/
      excluded_samples.csv
      summary.json
    checkpoints/
      last.pt
      best_train_loss.pt
  embeddings/
    sample_roles.csv
    embeddings_projector_64D.csv
    embeddings_preprojector_256D.csv
    embedding_qc.csv
    export_config.json
```

`sample_roles_original.csv` preserves the supplied split; `sample_roles.csv`
contains effective train IDs plus the unchanged validation IDs. `excluded_samples.csv`
has a header even when no samples are excluded. It records sample ID, original
role, reason, valid-pair count, required count and source file.
`best_train_loss.pt` is an auxiliary checkpoint selected on SSL training loss;
the default export uses `last.pt`, preserving the declared fixed-budget protocol.
These are model checkpoints, not full optimizer/RNG resume snapshots.
If fewer than two training samples remain, training stops after saving exclusion
and role manifests. No validation sample is moved into training.

Run one fold on the server, after activating the environment:

```bash
bash scripts/run_fold.sh 01 /server/raw_reads /server/results/ordna_seed0 /server/ordna_cache 150 200 20 --batch-size 8 --device cuda --seed 0
```

Replace sequence length150 and the illustrative update budget with the agreed
settings. `PYTHON_BIN=/path/to/python` can select an explicit environment interpreter.
