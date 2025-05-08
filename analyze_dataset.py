import pandas as pd
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

# --- Config ---
CSV_PATH = Path("wikiart_labels_imageonly_filtered.csv")
# CSV_PATH = Path("wikiart_labels_imageonly.csv")
SAVE_PLOT = True
PLOT_PATH = Path("outputs/emotion_distribution_filtered.png")
# PLOT_PATH = Path("outputs/emotion_distribution.png")

# --- Load dataset ---
df = pd.read_csv(CSV_PATH)

# --- Count emotions ---
all_emotions = []
for row in df["emotions"].fillna(""):
    all_emotions.extend([e.strip() for e in row.split(";") if e.strip()])

counter = Counter(all_emotions)

# --- Print distribution ---
print("Top 20 emotions by frequency:")
for emotion, count in counter.most_common(20):
    print(f"{emotion:20s}: {count}")

print(f"\nTotal unique emotions: {len(counter)}")

# --- Plot (optional) ---
if SAVE_PLOT:
    emotions, counts = zip(*counter.most_common())
    plt.figure(figsize=(12, 6))
    plt.bar(emotions, counts)
    plt.xticks(rotation=90)
    plt.ylabel("Number of images")
    # plt.title("Emotion Distribution in WikiArt Emotions Dataset")
    plt.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_PATH)
    print(f"Saved distribution plot to {PLOT_PATH}")

# --- Suggestions ---
rare = [e for e, c in counter.items() if c < 50]
print(f"\nEmotions with <50 images: {len(rare)}")
if rare:
    print("Example rare emotions:", rare[:10])
