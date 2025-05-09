#!/usr/bin/env bash
set -euo pipefail

# Folder where checkpoints & logs go
OUT_DIR="outputs_dataaug"

# Timestamped log-file
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

# ─── Run ───
python -u data_augmentation_normalized.py \
  --csv ./wikiart_labels_imageonly.csv \
  --images ./images/ \
  --output "$OUT_DIR" \
  --epochs 20 \
  --batch_size 32 \
  | tee "$LOG_FILE"

# Alternative Example:
# python -u data_augmentation.py \
#   --csv ./splits/train_filtered.csv \
#   --images ./images/ \
#   --output "$OUT_DIR" \
#   --epochs 20 \
#   --batch_size 32 \
#   | tee "$LOG_FILE"
