#!/bin/bash
#OAR -n ordna-embeddings
#OAR -l /nodes=1/gpu=1,walltime=24:00:00
#OAR --stdout ordna-embeddings.%jobid%.out
#OAR --stderr ordna-embeddings.%jobid%.err
#OAR --project pr-qiepb

set -e

# Repository contenente il codice ORDNA.
REPO_DIR="/bettik/PROJECTS/pr-qiepb/lampertl/ORDNA_WP3"

# Letture grezze: un CSV per campione, colonne Forward e Reverse.
RAW_DIR="/bettik/PROJECTS/pr-qiepb/lampertl/all"

# Split del fold 01.
SPLIT_FILE="$REPO_DIR/splits/strict_5km_3fold/index_mpa_merged_3_fold_01.csv"

# Checkpoint del training già completato.
CHECKPOINT="/bettik/PROJECTS/pr-qiepb/lampertl/runs_ordna_strict5km_3fold/seed0/fold_01/training/checkpoints/last.pt"

# Nuova cartella degli embedding e cache delle letture.
OUTPUT_DIR="/bettik/PROJECTS/pr-qiepb/lampertl/runs_ordna_strict5km_3fold/seed0/fold_01/embeddings_resumable"
CACHE_DIR="/bettik/PROJECTS/pr-qiepb/lampertl/ordna_token_cache"

# Ambiente.
source /applis/environments/conda.sh
conda activate zioboia

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Checkpoint: $CHECKPOINT"
echo "Embedding: $OUTPUT_DIR"

# Solo generazione degli embedding, senza training.
# --resume salta i campioni già completati da questo nuovo exporter.
python -u -m ordna_training.cli export \
  --samples-dir "$RAW_DIR" \
  --split "$SPLIT_FILE" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT_DIR" \
  --cache-dir "$CACHE_DIR" \
  --device cuda \
  --seed 0 \
  --resume
