!/usr/bin/env bash
python evaluate.py \
  --csv wikiart_labels_imageonly.csv \
  --images images \
  --checkpoint outputs/best_model.pth

# python evaluate.py \
#   --csv wikiart_labels_imageonly_filtered.csv \
#   --images images \
#   --checkpoint outputs/best_model.pth \
#   --tune_thresholds
