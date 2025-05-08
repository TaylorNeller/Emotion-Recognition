import pandas as pd

EMOTION_PREFIX = "ImageOnly: "          # << changed
INPUT_EMOTIONS_PATH = "WikiArt-Emotions-Ag4.tsv"
INPUT_INFO_PATH     = "WikiArt-info.tsv"
OUTPUT_PATH         = "wikiart_labels_imageonly.csv"

# 1. Load data
emotions_df = pd.read_csv(INPUT_EMOTIONS_PATH, sep="\t")
info_df     = pd.read_csv(INPUT_INFO_PATH,     sep="\t")

# 2. Attach image URLs
merged_df = pd.merge(
    emotions_df,
    info_df[["ID", "Image URL"]],
    on="ID",
    how="inner"
)

# 3. Pick the ImageOnly emotion columns
emotion_cols = [c for c in merged_df.columns if c.startswith(EMOTION_PREFIX)]
if len(emotion_cols) != 20:            # sanity check
    raise ValueError(
        f"Expected 20 emotion columns with prefix '{EMOTION_PREFIX}', "
        f"found {len(emotion_cols)}."
    )

# 4. Clean column labels -> 'anger', 'happiness', ...
cleaned_emotion_labels = [c.replace(EMOTION_PREFIX, "") for c in emotion_cols]
emotion_map = dict(zip(emotion_cols, cleaned_emotion_labels))

# 5. Build a semicolon-separated multi-label column
def extract_labels(row):
    return ";".join(
        emotion_map[col] for col in emotion_cols if row[col] == 1
    )

merged_df["emotions"] = merged_df.apply(extract_labels, axis=1)

# 6. Keep only essentials
final_df = (
    merged_df.rename(columns={"ID": "id", "Image URL": "image_url"})
    [["id", "image_url", "emotions"]]
    .query("emotions != ''")           # drop rows with no ImageOnly emotions
    .reset_index(drop=True)
)

# 7. Save
final_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved {len(final_df):,} rows to {OUTPUT_PATH}")
