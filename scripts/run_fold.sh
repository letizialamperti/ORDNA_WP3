#!/bin/bash
#OAR -n ordna-single-fold
#OAR -l /nodes=1/gpu=1,walltime=24:00:00
#OAR --stdout ordna.%jobid%.out
#OAR --stderr ordna.%jobid%.err
#OAR --project pr-qiepb

set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source "${RUN_CONFIG:-config/server.env}"

FOLD="${1:-$FOLD}"
case "$FOLD" in 01|02|03) ;; *) echo 'Fold must be 01, 02 or 03' >&2; exit 1 ;; esac

source "$CONDA_SH"
conda activate "$CONDA_ENV"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

SPLIT="splits/strict_5km_3fold/index_mpa_merged_3_fold_${FOLD}.csv"
RUN_DIR="$OUTPUT_ROOT/fold_${FOLD}"

"${PYTHON_BIN:-python}" -u -m ordna_training.cli train \
  --samples-dir "$RAW_DIR" --split "$SPLIT" \
  --output "$RUN_DIR/training" --cache-dir "$CACHE_DIR" \
  --sequence-length "$SEQUENCE_LENGTH" --subset-size "$SUBSET_SIZE" \
  --batch-size "$BATCH_SIZE" --steps-per-epoch "$STEPS_PER_EPOCH" --epochs "$EPOCHS" \
  --embedding-dim "$EMBEDDING_DIM" --representation-dim "$REPRESENTATION_DIM" \
  --device "$DEVICE" --seed "$SEED"

EXPORT_OPTIONS=()
if [[ -n "${MAX_EXPORT_SUBSETS:-}" ]]; then
  EXPORT_OPTIONS=(--max-subsets "$MAX_EXPORT_SUBSETS")
fi
"${PYTHON_BIN:-python}" -u -m ordna_training.cli export \
  --samples-dir "$RAW_DIR" --split "$SPLIT" \
  --checkpoint "$RUN_DIR/training/checkpoints/last.pt" \
  --output "$RUN_DIR/embeddings" --cache-dir "$CACHE_DIR" \
  --device "$DEVICE" --seed "$EXPORT_SEED" "${EXPORT_OPTIONS[@]}"
