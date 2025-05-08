import argparse
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import f1_score, average_precision_score


def parse_args():
    parser = argparse.ArgumentParser(description="Train a multi‑label emotion classifier on WikiArt images.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to wikiart_labels_imageonly.csv")
    parser.add_argument("--images", type=Path, required=True, help="Directory with images named <id>.jpg")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Directory to store logs and checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gp, self.gn, self.clip, self.eps = gamma_pos, gamma_neg, clip, eps

    def forward(self, logits, targets):
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        xs_pos = xs_pos.clamp(self.eps, 1.0)
        xs_neg = xs_neg.clamp(self.eps, 1.0)
        # basic BCE
        loss = targets * torch.log(xs_pos) + (1 - targets) * torch.log(xs_neg)
        # focal modulation
        loss *= (1 - xs_pos) ** self.gp * targets + xs_pos ** self.gn * (1 - targets)
        if self.clip:
            loss = torch.clamp(loss, max=-self.eps)  # small clip to avoid −inf
        return -loss.mean()


class WikiArtEmotionsDataset(Dataset):
    """Dataset that reads images by id and returns multi‑hot encoded emotion labels."""

    def __init__(self, csv_file: Path, images_dir: Path, transform=None, emotions=None):
        self.df = pd.read_csv(csv_file)
        if emotions is None:
            # Build full emotion vocabulary in sorted order for reproducible indices
            all_emotions = set()
            self.df["emotions"].fillna("", inplace=True)
            for row in self.df["emotions"]:
                all_emotions.update([e.strip() for e in row.split(";") if e.strip()])
            self.emotions = sorted(list(all_emotions))
        else:
            self.emotions = emotions
        self.emotion_to_idx = {e: i for i, e in enumerate(self.emotions)}
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _encode_emotions(self, emotion_str: str):
        vec = np.zeros(len(self.emotions), dtype=np.float32)
        for e in emotion_str.split(";"):
            e = e.strip()
            if e:
                vec[self.emotion_to_idx[e]] = 1.0
        return vec

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id"]
        img_path = self.images_dir / f"{img_id}.jpg"
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.from_numpy(self._encode_emotions(row["emotions"]))
        return image, label


def build_dataloaders(csv_path: Path, images_path: Path,
                      batch_size: int = 32,
                      val_split: float = 0.2,
                      seed: int = 42,
                      emotions=None):
    # ── Transforms ──────────────────────────────────────────────────────────────
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ── Make a *base* dataset (just to get class list) ─────────────────────────
    base_ds = WikiArtEmotionsDataset(csv_path, images_path, transform=None)
    if emotions is None:            # we were not given a list → build one
        emotions = base_ds.emotions

    # ── Train / val index split ────────────────────────────────────────────────
    num_val = int(len(base_ds) * val_split)
    num_train = len(base_ds) - num_val
    g = torch.Generator().manual_seed(seed)
    train_idx, val_idx = torch.utils.data.random_split(
        range(len(base_ds)), [num_train, num_val], generator=g
    )

    # ── Two independent datasets with *different* transforms ──────────────────
    train_ds = Subset(
        WikiArtEmotionsDataset(csv_path, images_path,
                               transform=train_tfms, emotions=emotions),
        train_idx.indices if hasattr(train_idx, "indices") else train_idx
    )
    val_ds = Subset(
        WikiArtEmotionsDataset(csv_path, images_path,
                               transform=val_tfms, emotions=emotions),
        val_idx.indices if hasattr(val_idx, "indices") else val_idx
    )

    # ── Class-balanced sampler built *only* from the train subset ─────────────
    weights = []
    for _, lbl in train_ds:
        # 1 / (#active emotions in that sample)
        weights.append(1.0 / (lbl.sum().item() or 1))
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds),
                                    replacement=True)

    # ── Dataloaders ────────────────────────────────────────────────────────────
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds,   batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, emotions

def build_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def calculate_metrics(outputs, targets):
    """Returns micro‑F1, macro‑F1, and mAP (macro average precision)."""
    probs = torch.sigmoid(outputs).cpu().numpy()
    true = targets.cpu().numpy()
    bin_preds = (probs >= 0.5).astype(int)
    micro_f1 = f1_score(true.ravel(), bin_preds.ravel(), average="micro", zero_division=0)
    macro_f1 = f1_score(true, bin_preds, average="macro", zero_division=0)
    mAP = average_precision_score(true, probs, average="macro")
    return micro_f1, macro_f1, mAP


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_outputs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            all_outputs.append(outputs)
            all_labels.append(labels)
    outputs_cat = torch.cat(all_outputs)
    labels_cat = torch.cat(all_labels)
    micro_f1, macro_f1, mAP = calculate_metrics(outputs_cat, labels_cat)
    return running_loss / len(loader.dataset), micro_f1, macro_f1, mAP

def collect_emotions(csv_paths):
    all_emotions = set()
    for p in csv_paths:
        df = pd.read_csv(p)
        df["emotions"].fillna("", inplace=True)
        for cell in df["emotions"]:
            all_emotions.update(e.strip() for e in cell.split(";") if e.strip())
    return sorted(all_emotions)

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv = "splits/train.csv"
    val_csv   = "splits/val.csv"
    test_csv  = "splits/test.csv"   # <- for later evaluation

    emotions = collect_emotions([train_csv, val_csv, test_csv])

    # Read the pre-made splits
    train_loader,  _ , _ = build_dataloaders(train_csv, args.images,
                                                    args.batch_size, 0.0,  # val split already fixed
                                                    args.seed,
                                                    emotions=emotions)
    val_loader,    _ , _        = build_dataloaders(val_csv,   args.images,
                                                    args.batch_size, 0.0,
                                                    args.seed,
                                                    emotions=emotions)

    # train_loader, val_loader, emotions = build_dataloaders(args.csv, args.images, args.batch_size, args.val_split, args.seed)
    num_classes = len(emotions)
    print(f"Detected {num_classes} emotion classes: {emotions}")

    # Compute class positive weights to address imbalance
    class_counts = np.zeros(num_classes)
    for _, labels in train_loader.dataset:
        class_counts += labels.numpy()
    pos_weights = (class_counts.sum() - class_counts) / (class_counts + 0.000001)
    pos_weights = torch.tensor(pos_weights, dtype=torch.float32).to(device)

    model = build_model(num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    # criterion = AsymmetricLoss()      # to combat class imbalance
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_map = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, micro_f1, macro_f1, mAP = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{args.epochs} » train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | microF1: {micro_f1:.4f} | macroF1: {macro_f1:.4f} | mAP: {mAP:.4f}"
        )
        # Save best model by mAP
        if mAP > best_map:
            best_map = mAP
            ckpt_path = args.output / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mAP": mAP,
                "emotions": emotions,
            }, ckpt_path)
            print(f"→ New best model saved to {ckpt_path} (mAP={mAP:.4f})")

    # Save final model after training completes
    final_path = args.output / "last_model.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Training finished. Final model weights saved to {final_path}")


if __name__ == "__main__":
    main()
