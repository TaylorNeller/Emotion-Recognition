#!/usr/bin/env python
"""
evaluate.py – Evaluate a trained WikiArt multi-label emotion model
         *with optional per-class threshold tuning* against a
         pre-made train/val/test split.

Example
-------
# plain evaluation (fixed 0.5 threshold)
python evaluate.py \
  --val_csv   splits/val.csv \
  --test_csv  splits/test.csv \
  --images    images \
  --checkpoint outputs/best_model.pth

# tune thresholds on the validation set first
python evaluate.py \
  --val_csv   splits/val.csv \
  --test_csv  splits/test.csv \
  --images    images \
  --checkpoint outputs/best_model.pth \
  --tune_thresholds
"""
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import torch
from torchvision import transforms
from sklearn.metrics import f1_score, average_precision_score

# --- Local imports -----------------------------------------------------------
from wikiart_emotions_cnn import (
    WikiArtEmotionsDataset,
    build_model,
)

# ─────────────────────────────────────────────────────────────────────────────
VAL_TFMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def tune_thresholds(probs: np.ndarray, labels: np.ndarray, steps: int = 20):
    """Return per-class thresholds maximising F1 on *this* set."""
    C = labels.shape[1]
    thresholds = np.full(C, 0.5, dtype=np.float32)
    for c in range(C):
        best_f1, best_t = 0.0, 0.5
        for t in np.linspace(0.05, 0.95, steps):
            preds = (probs[:, c] >= t).astype(int)
            f1 = f1_score(labels[:, c], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
    return thresholds


def calculate_metrics(probs: torch.Tensor,
                      labels: torch.Tensor,
                      thresholds: np.ndarray | float = 0.5):
    """Compute micro-F1, macro-F1, mAP and per-class AP."""
    probs_np  = probs.cpu().numpy()
    labels_np = labels.cpu().numpy()

    if np.isscalar(thresholds):
        bin_preds = (probs_np >= thresholds).astype(int)
    else:
        bin_preds = (probs_np >= thresholds[None, :]).astype(int)

    micro_f1 = f1_score(labels_np.ravel(), bin_preds.ravel(),
                        average="micro", zero_division=0)
    macro_f1 = f1_score(labels_np, bin_preds,
                        average="macro", zero_division=0)
    mAP = average_precision_score(labels_np, probs_np, average="macro")
    per_class_ap = average_precision_score(labels_np, probs_np, average=None)
    return micro_f1, macro_f1, mAP, per_class_ap


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate WikiArt emotion model")
    p.add_argument("--val_csv",   type=Path, required=True,
                   help="CSV that defines the *validation* split (for threshold tuning)")
    p.add_argument("--test_csv",  type=Path, required=True,
                   help="CSV that defines the *test* split (final scoring)")
    p.add_argument("--images",    type=Path, required=True,
                   help="Directory with <id>.jpg files")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to best_model.pth or last_model.pth")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--tune_thresholds", action="store_true",
                   help="Optimise a separate threshold for every emotion on the "
                        "validation split before scoring the test set")
    return p.parse_args()


def make_loader(csv_path: Path, images_dir: Path,
                emotions: list[str], batch_size: int):
    ds = WikiArtEmotionsDataset(csv_path, images_dir,
                                transform=VAL_TFMS, emotions=emotions)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint ──────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    emotions = ckpt.get("emotions")
    if emotions is None:  # fallback if checkpoint doesn't store vocab
        base_ds = WikiArtEmotionsDataset(args.val_csv, args.images, transform=None)
        emotions = base_ds.emotions
    num_classes = len(emotions)

    # ── Model ───────────────────────────────────────────────────────────────
    model = build_model(num_classes).to(device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # ── Data loaders (fixed splits) ─────────────────────────────────────────
    val_loader  = make_loader(args.val_csv,  args.images, emotions, args.batch_size)
    test_loader = make_loader(args.test_csv, args.images, emotions, args.batch_size)

    # ── Threshold tuning (optional, done on *val* split) ────────────────────
    thresholds = 0.5
    if args.tune_thresholds:
        print("Tuning per-class thresholds on validation split …", flush=True)
        all_probs, all_labels = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                probs = torch.sigmoid(model(imgs))
                all_probs.append(probs.cpu())
                all_labels.append(lbls)
        probs_val  = torch.cat(all_probs)
        labels_val = torch.cat(all_labels)
        thresholds = tune_thresholds(probs_val.numpy(), labels_val.numpy())
        print("done.\n")

    # ── Final evaluation on *test* split ────────────────────────────────────
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            probs = torch.sigmoid(model(imgs))
            all_probs.append(probs.cpu())
            all_labels.append(lbls)

    probs_test  = torch.cat(all_probs)
    labels_test = torch.cat(all_labels)

    μF1, M_F1, mAP, per_cls_AP = calculate_metrics(
        probs_test, labels_test, thresholds)

    # ── Report ──────────────────────────────────────────────────────────────
    print("Evaluation on held-out test split")
    print("══════════════════════════════════")
    if isinstance(thresholds, np.ndarray):
        print("† custom per-class thresholds applied\n")
    print(f"micro-F1 : {μF1:6.4f}")
    print(f"macro-F1 : {M_F1:6.4f}")
    print(f"mAP      : {mAP:6.4f}\n")

    w = max(len(e) for e in emotions)
    print("Per-class AP (threshold-free ranking)")
    print("-------------------------------------")
    for e, ap in zip(emotions, per_cls_AP):
        print(f"{e:<{w}} : {ap:6.4f}")

    if isinstance(thresholds, np.ndarray):
        print("\nOptimal thresholds (found on validation split)")
        print("---------------------------------------------")
        for e, t in zip(emotions, thresholds):
            print(f"{e:<{w}} : {t:4.2f}")


if __name__ == "__main__":
    main()
