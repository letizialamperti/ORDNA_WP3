#!/bin/bash
#OAR -n ordna-single-fold
#OAR -l /nodes=1/gpu=1/core=12,walltime=24:00:00
#OAR -p gpumodel='A100'
#OAR --stdout ordna.%jobid%.out
#OAR --stderr ordna.%jobid%.err
#OAR --project pr-qiepb

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
  echo "OAR: configure config/server.env, then oarsub -S scripts/run_fold.sh"
  echo "Or: bash scripts/run_fold.sh FOLD RAW_DIR OUTPUT_ROOT CACHE_DIR SEQUENCE_LENGTH STEPS_PER_EPOCH EPOCHS [training options]"
  echo "RUN_CONFIG selects another environment file. CONDA_ENV='' skips Conda activation."
  exit 0
fi

config_file="${RUN_CONFIG:-$repo_root/config/server.env}"
if [[ -f "$config_file" ]]; then
  source "$config_file"
fi

if [[ $# -ge 7 ]]; then
  FOLD="$1"
  RAW_DIR="$2"
  OUTPUT_ROOT="$3"
  CACHE_DIR="$4"
  SEQUENCE_LENGTH="$5"
  STEPS_PER_EPOCH="$6"
  EPOCHS="$7"
  shift 7
elif [[ $# -ne 0 ]]; then
  echo "Expected zero arguments with config/server.env, or seven positional arguments; use --help." >&2
  exit 2
fi

: "${FOLD:?Set FOLD in config/server.env}"
: "${RAW_DIR:?Set RAW_DIR in config/server.env}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT in config/server.env}"
: "${CACHE_DIR:?Set CACHE_DIR in config/server.env}"
: "${SEQUENCE_LENGTH:?Set SEQUENCE_LENGTH in config/server.env}"
: "${STEPS_PER_EPOCH:?Set STEPS_PER_EPOCH in config/server.env}"
: "${EPOCHS:?Set EPOCHS in config/server.env}"

case "$FOLD" in
  01|02|03) ;;
  *)
    echo "FOLD must be 01, 02 or 03" >&2
    exit 2
    ;;
esac

[[ -d "$RAW_DIR" ]] || {
  echo "Missing raw directory: $RAW_DIR" >&2
  exit 2
}

# Attivazione dell'ambiente Conda.
CONDA_ENV="${CONDA_ENV-ordna-training}"

if [[ -n "$CONDA_ENV" ]]; then
  source "${CONDA_SH:-/applis/environments/conda.sh}"
  conda activate "$CONDA_ENV"
fi

python_bin="${PYTHON_BIN:-python}"

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

split="$repo_root/splits/strict_5km_3fold/index_mpa_merged_3_fold_${FOLD}.csv"
device="${DEVICE:-cuda}"

echo "OAR job: ${OAR_JOB_ID:-manual}; fold: $FOLD; device: $device"
echo "Split: $split"
echo "Run directory: $OUTPUT_ROOT/fold_${FOLD}"

# Training: soltanto i campioni train dello split.
"$python_bin" -m ordna_training.cli train \
  --samples-dir "$RAW_DIR" \
  --split "$split" \
  --output "$OUTPUT_ROOT/fold_${FOLD}/training" \
  --cache-dir "$CACHE_DIR" \
  --sequence-length "$SEQUENCE_LENGTH" \
  --steps-per-epoch "$STEPS_PER_EPOCH" \
  --epochs "$EPOCHS" \
  --batch-size "${BATCH_SIZE:-8}" \
  --subset-size "${SUBSET_SIZE:-1000}" \
  --embedding-dim "${EMBEDDING_DIM:-64}" \
  --representation-dim "${REPRESENTATION_DIM:-256}" \
  --device "$device" \
  --seed "${SEED:-0}" \
  "$@"

# Esportazione degli embedding con il modello congelato.
export_options=()

if [[ -n "${MAX_EXPORT_SUBSETS:-}" ]]; then
  export_options+=(--max-subsets "$MAX_EXPORT_SUBSETS")
fi

"$python_bin" -m ordna_training.cli export \
  --samples-dir "$RAW_DIR" \
  --split "$split" \
  --checkpoint "$OUTPUT_ROOT/fold_${FOLD}/training/checkpoints/last.pt" \
  --output "$OUTPUT_ROOT/fold_${FOLD}/embeddings" \
  --cache-dir "$CACHE_DIR" \
  --device "$device" \
  --seed "${EXPORT_SEED:-0}" \
  "${export_options[@]}"
