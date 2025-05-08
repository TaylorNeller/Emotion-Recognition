import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import f1_score, average_precision_score
from tqdm.auto import tqdm

# We prefer timm ViT implementation for flexibility / pretrained weights
try:
    import timm
except ImportError as e:
    raise ImportError("timm is required: pip install timm")


class WikiArtEmotionsDataset(Dataset):
    """Dataset that serves images + multi-hot emotion vectors."""

    def __init__(self, csv_file: Path, images_dir: Path, transform=None, emotions=None):
        self.df = pd.read_csv(csv_file)
        if emotions is None:
            vocab = set()
            self.df["emotions"].fillna("", inplace=True)
            for row in self.df["emotions"]:
                vocab.update([e.strip() for e in row.split(";") if e.strip()])
            self.emotions = sorted(vocab)
        else:
            self.emotions = emotions
        self.emotion_to_idx = {e: i for i, e in enumerate(self.emotions)}
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _encode(self, emotion_str):
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
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.from_numpy(self._encode(row["emotions"]))
        return img, label


def build_dataloaders(csv, images, batch_size, val_split, seed, img_size=224):
    # ViT likes simple random resized crop (same aspect ratio variety) and normalization
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    full_ds = WikiArtEmotionsDataset(csv, images, train_tfms)
    val_len = int(len(full_ds) * val_split)
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len], generator=torch.Generator().manual_seed(seed))
    val_ds.dataset.transform = val_tfms

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, full_ds.emotions


def build_model(num_classes, img_size=224):
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=num_classes,
        img_size=img_size,
    )
    # Replace head to suit multi-label (linear only, but timm already set)
    return model


def calc_metrics(outputs, targets):
    probs = torch.sigmoid(outputs).cpu().numpy()
    true = targets.cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    micro = f1_score(true.ravel(), preds.ravel(), average="micro", zero_division=0)
    macro = f1_score(true, preds, average="macro", zero_division=0)
    mAP = average_precision_score(true, probs, average="macro")
    return micro, macro, mAP


class AsymmetricLoss(nn.Module):
    """Focal-like loss that balances positive/negative examples without huge pos_weights."""

    def __init__(self, gamma_neg=4, gamma_pos=0, eps=1e-8):
        super().__init__()
        self.gn = gamma_neg
        self.gp = gamma_pos
        self.eps = eps

    def forward(self, logits, targets):
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        xs_pos = xs_pos.clamp(self.eps, 1.0)
        xs_neg = xs_neg.clamp(self.eps, 1.0)
        loss = targets * torch.log(xs_pos) + (1 - targets) * torch.log(xs_neg)
        loss *= (1 - xs_pos) ** self.gp * targets + xs_pos ** self.gn * (1 - targets)
        return -loss.mean()


def parse_args():
    p = argparse.ArgumentParser("Train ViT on WikiArt emotions (multi‑label)")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs_vit"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img_size", type=int, default=224, help="Input resolution for ViT (multiple of 16)")
    p.add_argument("--no_progress", action="store_true")
    return p.parse_args()


def train_epoch(model, loader, criterion, optim_, device, epoch, show):
    model.train()
    total = 0.0
    iterator = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False) if show else loader
    for imgs, labels in iterator:
        imgs, labels = imgs.to(device), labels.to(device)
        optim_.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optim_.step()
        total += loss.item() * imgs.size(0)
        if show:
            iterator.set_postfix(loss=f"{loss.item():.4f}")
    return total / len(loader.dataset)


def evaluate(model, loader, criterion, device, show):
    model.eval()
    total = 0.0
    outs, tgts = [], []
    iterator = tqdm(loader, desc="Validation", leave=False) if show else loader
    with torch.no_grad():
        for imgs, labels in iterator:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            total += loss.item() * imgs.size(0)
            outs.append(out)
            tgts.append(labels)
    outs = torch.cat(outs)
    tgts = torch.cat(tgts)
    micro, macro, mAP = calc_metrics(outs, tgts)
    return total / len(loader.dataset), micro, macro, mAP


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_loader, val_loader, emotions = build_dataloaders(
        args.csv, args.images, args.batch_size, args.val_split, args.seed, args.img_size
    )
    num_cls = len(emotions)
    print(f"Classes: {num_cls} — {emotions}")

    model = build_model(num_cls, args.img_size).to(device)
    criterion = AsymmetricLoss()
    optim_ = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_map = 0.0
    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, tr_loader, criterion, optim_, device, ep, not args.no_progress)
        val_loss, mic_f1, mac_f1, mAP = evaluate(model, val_loader, criterion, device, not args.no_progress)
        print(
            f"Epoch {ep}/{args.epochs} | train_loss {tr_loss:.4f} | val_loss {val_loss:.4f} | "
            f"microF1 {mic_f1:.4f} | macroF1 {mac_f1:.4f} | mAP {mAP:.4f}",
            flush=True,
        )
        if mAP > best_map:
            best_map = mAP
            ckpt = {
                "epoch": ep,
                "state_dict": model.state_dict(),
                "mAP": mAP,
                "emotions": emotions,
            }
            torch.save(ckpt, args.output / "best_vit.pth")
            print("→ Best model updated (mAP {:.4f})".format(mAP), flush=True)

    # final save
    torch.save(model.state_dict(), args.output / "last_vit.pth")
    print("Training complete. Weights saved in", args.output.resolve())


if __name__ == "__main__":
    main()
