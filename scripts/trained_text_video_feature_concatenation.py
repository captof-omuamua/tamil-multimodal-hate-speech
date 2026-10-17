import pandas as pd
import numpy as np
import os

# Define file paths
excel_path = "/home/raptor/Desktop/sem_7/project_phase_1/dataset/multimodal-hate/training_data/train_embeddings.xlsx"  # The uploaded text embedding file
npy_dir = "/home/raptor/Desktop/sem_7/project_phase_1/dataset/multimodal-hate/training_data/vit_train/avg_frame_wise_features"
output_excel_path = "/home/raptor/Desktop/sem_7/project_phase_1/dataset/multimodal-hate/training_data/synthetic_data/Concatenated_Train_Embeddings.xlsx"

# Load text embeddings from Excel
df = pd.read_excel(excel_path, usecols="A", nrows=40, header=None)  # Load 40 text embeddings

# Prepare output data storage
concatenated_data = []

# Process each text embedding and concatenate with corresponding video feature
for i in range(40):
    # Load the text embedding for this row
    text_embedding = np.fromstring(df.iloc[i, 0].strip("[]"), sep=',')
    
    # Load corresponding video embedding
    npy_filename = f"TAM_{str(i+1).zfill(2)}_avg_features.npy"
    video_embedding = np.load(os.path.join(npy_dir, npy_filename))
    
    # Concatenate text and video embeddings
    synthetic_feature = np.concatenate([text_embedding, video_embedding])
    concatenated_data.append(synthetic_feature)

# Save the concatenated data to a new Excel file
concatenated_df = pd.DataFrame(concatenated_data)
concatenated_df.to_excel(output_excel_path, index=False, header=False)

print(f"Concatenated features saved to {output_excel_path}")
