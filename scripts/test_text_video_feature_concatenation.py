import pandas as pd
import numpy as np
import os

# Define file paths
excel_path = "/home/raptor/Desktop/sem_7/project_phase_1/dataset/multimodal-hate/test_data/synthetic_data/test_data_embeddings.xlsx" # Uploaded text embeddings file
npy_dir = "/home/raptor/Desktop/sem_7/project_phase_1/dataset/multimodal-hate/test_data/vit_test/avg_frame_wise_features"
output_excel_path = "/home/raptor/Desktop/sem_7/project_phase_1/dataset/multimodal-hate/test_data/synthetic_data/Concatenated_Test_Data.xlsx"

# Load text embeddings from Excel
df = pd.read_excel(excel_path, usecols="A", nrows=12, header=None)  # Adjust as per the actual data

# Prepare output data storage
concatenated_data = []

# Loop through each text embedding and its corresponding video embedding
for i in range(12):
    # Read the text embedding for this row
    try:
        text_embedding = np.fromstring(df.iloc[i, 0].strip("[]"), sep=',')
        if text_embedding.shape[0] != 768:  # Ensure it’s a 1x768 embedding
            print(f"Warning: Row {i+1} has an unexpected size of {text_embedding.shape[0]}. Skipping.")
            continue
    except ValueError as e:
        print(f"Error parsing row {i+1}: {e}. Skipping.")
        continue

    # Load the corresponding video embedding
    npy_filename = f"TAM_{str(42 + i).zfill(2)}_avg_features.npy"
    video_embedding = np.load(os.path.join(npy_dir, npy_filename))

    # Ensure the video embedding is also 1x768
    if video_embedding.shape[0] != 768:
        print(f"Warning: Video embedding from {npy_filename} has an unexpected size {video_embedding.shape[0]}. Skipping.")
        continue

    # Concatenate text and video embeddings
    synthetic_feature = np.concatenate([text_embedding, video_embedding])
    concatenated_data.append(synthetic_feature)

# Save the concatenated data to a new Excel file
concatenated_df = pd.DataFrame(concatenated_data)
concatenated_df.to_excel(output_excel_path, index=False, header=False)

print(f"Concatenated features saved to {output_excel_path}")
