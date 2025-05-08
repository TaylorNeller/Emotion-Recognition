#!/usr/bin/env bash
set -euo pipefail

# Folder where checkpoints & logs go
OUT_DIR="vit_outputs"

# Timestamped log-file
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

# ─── Run ───
# python -u wikiart_emotions_vit.py \
#   --csv ./wikiart_labels_imageonly_filtered.csv \
#   --images ./images/ \
#   --output "$OUT_DIR" \
#   --epochs 20 \
#   --batch_size 16 \
#   | tee "$LOG_FILE"

python -u wikiart_emotions_vit2.py \
  --csv ./wikiart_labels_imageonly.csv \
  --images ./images/ \
  --output "$OUT_DIR" \
  --epochs 20 \
  --batch_size 16 \
  | tee "$LOG_FILE"