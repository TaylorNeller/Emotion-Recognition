import pandas as pd

# List of emotions to filter out
unwanted_emotions = {
    'anger',
    'disgust',
    'gratitude',
    'pessimism',
    'shame',
    'regret',
    'disagreeableness',
    'agreeableness',
    'shyness',
    'neutral'
}

# Load the CSV
input_file = 'wikiart_labels_imageonly.csv'  # change to your input file path
output_file = 'wikiart_labels_imageonly_filtered.csv'
df = pd.read_csv(input_file)

# Function to check if any unwanted emotion is present
def contains_unwanted_emotion(emotion_str):
    emotions = [e.strip() for e in emotion_str.split(';')]
    return any(e in unwanted_emotions for e in emotions)

# Filter the DataFrame
filtered_df = df[~df['emotions'].apply(contains_unwanted_emotion)]

# Save the result
filtered_df.to_csv(output_file, index=False)

print(f"Filtered data saved to {output_file}.")
