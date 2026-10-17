import pandas as pd
import numpy as np
import os

# Define file paths
excel_path = "/home/raptor/Desktop/sem_7/project_phase_1/dataset/multimodal-hate/training_data/synthetic_data/Synthetic_embeddings_768.xlsx"
npy_dir = "/home/raptor/Desktop/sem_7/project_phase_1/dataset/multimodal-hate/training_data/vit_train/avg_frame_wise_features"
output_excel_path = "/home/raptor/Desktop/sem_7/project_phase_1/dataset/multimodal-hate/training_data/synthetic_data/Concatenated_Synthetic_Features.xlsx"

# Load text embeddings from Excel
df = pd.read_excel(excel_path, usecols="A", skiprows=1, nrows=2000, header=None)  # Load A2 to A2001

# Prepare output data storage
concatenated_data = []

# Process embeddings row by row and concatenate with corresponding video feature
for i in range(40):
    # Calculate the range for the current 50 text embeddings
    start_idx = i * 50
    end_idx = start_idx + 50

    # Load corresponding video embedding
    npy_filename = f"TAM_{str(i+1).zfill(2)}_avg_features.npy"
    video_embedding = np.load(os.path.join(npy_dir, npy_filename))

    # Concatenate each text embedding with the video embedding
    for j in range(start_idx, end_idx):
        text_embedding = np.fromstring(df.iloc[j, 0].strip("[]"), sep=',')
        synthetic_feature = np.concatenate([text_embedding, video_embedding])
        concatenated_data.append(synthetic_feature)

# Save the concatenated data to a new Excel file
concatenated_df = pd.DataFrame(concatenated_data)
concatenated_df.to_excel(output_excel_path, index=False, header=False)

print(f"Concatenated features saved to {output_excel_path}")
