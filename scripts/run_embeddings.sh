#!/bin/bash
#OAR -n ordna-embeddings
#OAR -l /nodes=1/gpu=1,walltime=24:00:00
#OAR --stdout ordna-embeddings.%jobid%.out
#OAR --stderr ordna-embeddings.%jobid%.err
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
CHECKPOINT="${CHECKPOINT_PATH:-$RUN_DIR/training/checkpoints/last.pt}"
# Separate destination from the old interrupted, non-resumable export.
EMBEDDINGS_DIR="${EMBEDDINGS_DIR:-$RUN_DIR/embeddings_resumable}"
[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }

EXPORT_OPTIONS=()
if [[ -n "${MAX_EXPORT_SUBSETS:-}" ]]; then
  EXPORT_OPTIONS=(--max-subsets "$MAX_EXPORT_SUBSETS")
fi
echo "Export checkpoint: $CHECKPOINT"
echo "Export directory: $EMBEDDINGS_DIR"
"${PYTHON_BIN:-python}" -u -m ordna_training.cli export \
  --samples-dir "$RAW_DIR" --split "$SPLIT" --checkpoint "$CHECKPOINT" \
  --output "$EMBEDDINGS_DIR" --cache-dir "$CACHE_DIR" \
  --device "$DEVICE" --seed "$EXPORT_SEED" --resume "${EXPORT_OPTIONS[@]}"
