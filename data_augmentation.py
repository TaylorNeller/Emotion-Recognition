import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.metrics import f1_score, average_precision_score

def parse_args():
    parser = argparse.ArgumentParser(description="Train with data augmentation using WikiArt dataset.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to the CSV file for training.")
    parser.add_argument("--images", type=Path, required=True, help="Path to the images directory.")
    parser.add_argument("--output", type=Path, default=Path("outputs_dataaug"), help="Output directory for logs and checkpoints.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

class WikiArtDataset(Dataset):
    def __init__(self, csv_file, images_dir, transform=None, emotions=None):
        self.df = pd.read_csv(csv_file)
        if emotions is None:
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

    def _encode_emotions(self, emotion_str):
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
        if self.transform:
            image = self.transform(image)
        label = torch.from_numpy(self._encode_emotions(row["emotions"]))
        return image, label

def build_dataloaders(csv_path, images_path, batch_size, val_split, seed, emotions=None):
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor()
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    base_ds = WikiArtDataset(csv_path, images_path, transform=None)
    if emotions is None:
        emotions = base_ds.emotions

    num_val = int(len(base_ds) * val_split)
    num_train = len(base_ds) - num_val
    g = torch.Generator().manual_seed(seed)
    train_idx, val_idx = torch.utils.data.random_split(range(len(base_ds)), [num_train, num_val], generator=g)

    train_ds = Subset(WikiArtDataset(csv_path, images_path, transform=train_tfms, emotions=emotions),
                      train_idx.indices if hasattr(train_idx, "indices") else train_idx)
    val_ds = Subset(WikiArtDataset(csv_path, images_path, transform=val_tfms, emotions=emotions),
                    val_idx.indices if hasattr(val_idx, "indices") else val_idx)

    weights = [1.0 / (lbl.sum().item() or 1) for _, lbl in train_ds]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, emotions

def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

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
    probs = torch.sigmoid(outputs_cat).cpu().numpy()
    true = labels_cat.cpu().numpy()
    bin_preds = (probs >= 0.5).astype(int)
    micro_f1 = f1_score(true.ravel(), bin_preds.ravel(), average="micro", zero_division=0)
    macro_f1 = f1_score(true, bin_preds, average="macro", zero_division=0)
    mAP = average_precision_score(true, probs, average="macro")
    return running_loss / len(loader.dataset), micro_f1, macro_f1, mAP

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, emotions = build_dataloaders(
        args.csv, args.images, args.batch_size, args.val_split, args.seed
    )
    num_classes = len(emotions)

    class_counts = np.zeros(num_classes)
    for _, labels in train_loader.dataset:
        class_counts += labels.numpy()
    pos_weights = (class_counts.sum() - class_counts) / (class_counts + 1e-6)
    pos_weights = torch.tensor(pos_weights, dtype=torch.float32).to(device)

    model = build_model(num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_map = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, micro_f1, macro_f1, mAP = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Micro F1: {micro_f1:.4f} | Macro F1: {macro_f1:.4f} | mAP: {mAP:.4f}"
        )
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
            print(f"Saved new best model to {ckpt_path} (mAP={mAP:.4f})")

    final_path = args.output / "last_model.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Final model saved to {final_path}")

if __name__ == "__main__":
    main()
