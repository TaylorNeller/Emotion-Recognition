import pandas as pd
import os
import requests
from tqdm import tqdm

# Load the cleaned CSV
df = pd.read_csv("wikiart_labels.csv")

# Output folder
os.makedirs("images", exist_ok=True)

# Loop through each row and download the image
for _, row in tqdm(df.iterrows(), total=len(df)):
    img_id = row['id']
    img_url = row['image_url']
    img_path = f"images/{img_id}.jpg"

    # Skip if already downloaded
    if os.path.exists(img_path):
        continue

    try:
        response = requests.get(img_url, timeout=10)
        if response.status_code == 200:
            with open(img_path, "wb") as f:
                f.write(response.content)
    except Exception as e:
        print(f"❌ Failed to download {img_id}: {e}")