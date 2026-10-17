import pandas as pd
import re

# Load the CSV file
input_file = "/home/raptor/Downloads/dataset (1).csv"  # Replace with your actual CSV file name
output_file = "/home/raptor/Desktop/sem_7/project_phase_1/dataset/multimodal-hate/training_data/synthetic_data/emoji_filtered_output.csv"  # The name of the output CSV file

# Regular expression pattern for matching emojis
emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251" 
    "]+",
    flags=re.UNICODE
)

# Read the CSV file into a DataFrame
df = pd.read_csv(input_file)

# Filter out rows with the label "Non-Hate-Speech"
filtered_df = df[df['label'] != 'Non-Hate-Speech']

# Function to remove emojis from text
def remove_emojis(text):
    return emoji_pattern.sub(r'', text)

# Apply the emoji removal function to the 'text' column
filtered_df['text'] = filtered_df['text'].apply(remove_emojis)

# Save the filtered and cleaned data to a new CSV file
filtered_df.to_csv(output_file, index=False)

print(f"Filtered and emoji-free data saved to {output_file}")
