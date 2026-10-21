#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 7 ]]; then
  echo "Usage: bash scripts/run_fold.sh FOLD RAW_DIR OUTPUT_ROOT CACHE_DIR SEQUENCE_LENGTH STEPS_PER_EPOCH EPOCHS [training options]" >&2
  exit 2
fi
fold="$1"; raw_dir="$2"; output_root="$3"; cache_dir="$4"
seq_len="$5"; steps="$6"; epochs="$7"; shift 7
case "$fold" in 01|02|03) ;; *) echo "FOLD must be 01, 02 or 03" >&2; exit 2 ;; esac
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
split="$repo_root/splits/strict_5km_3fold/index_mpa_merged_3_fold_${fold}.csv"
python_bin="${PYTHON_BIN:-python}"
"$python_bin" -m ordna_training.cli train --samples-dir "$raw_dir" --split "$split" \
  --output "$output_root/fold_${fold}/training" --cache-dir "$cache_dir" \
  --sequence-length "$seq_len" --steps-per-epoch "$steps" --epochs "$epochs" "$@"
"$python_bin" -m ordna_training.cli export --samples-dir "$raw_dir" --split "$split" \
  --checkpoint "$output_root/fold_${fold}/training/checkpoints/last.pt" \
  --output "$output_root/fold_${fold}/embeddings" --cache-dir "$cache_dir"
