# ───────────────────────────────── prepare_split.py ──────────────────────────
"""
Create a deterministic train/val/test split with the same ratios used in
script A (train=0.64, val=0.16, test=0.20) and save them to disk.

Example
-------
python prepare_split.py \
    --csv wikiart_labels_imageonly_filtered.csv \
    --out_dir splits \
    --seed 42
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def make_split(df: pd.DataFrame, seed: int = 42,
               test_ratio: float = 0.20,
               val_ratio: float = 0.20):  # ← of the remaining “train_val” set
    # first: carve out the final test set
    train_val_df, test_df = train_test_split(
        df, test_size=test_ratio, random_state=seed, shuffle=True
    )
    # second: split the remainder into train/val
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_ratio, random_state=seed, shuffle=True
    )
    return train_df.reset_index(drop=True), \
           val_df.reset_index(drop=True), \
           test_df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True,
                    help="Full WikiArt CSV (id, emotions, …).")
    ap.add_argument("--out_dir", type=Path, default=Path("splits"),
                    help="Directory where train/val/test CSVs will be written.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    train_df, val_df, test_df = make_split(df, seed=args.seed)

    train_df.to_csv(args.out_dir / "train.csv", index=False)
    val_df.to_csv(args.out_dir / "val.csv",   index=False)
    test_df.to_csv(args.out_dir / "test.csv", index=False)

    print(f"✓ Splits written to {args.out_dir.resolve()}")
    print(f"  train: {len(train_df)} rows\n"
          f"  val  : {len(val_df)} rows\n"
          f"  test : {len(test_df)} rows")


if __name__ == "__main__":
    main()
