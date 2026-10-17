import os
import numpy as np
import pandas as pd

# Directories for text embeddings and ViT16 video features
text_embeddings_dir = '/home/raptor/Desktop/sem_7/project_phase_1/dataset/multimodal-hate/training_data/text_embeddings'
vit16_features_path = '/home/raptor/Desktop/sem_7/project_phase_1/dataset/multimodal-hate/training_data/vit_train/avg_video_features_vit.xlsx'
output_npy_dir = '/home/raptor/Desktop/sem_7/project_phase_1/dataset/multimodal-hate/training_data/concatenated_features_npy'  # Directory to save concatenated .npy files
output_excel_path = '/home/raptor/Desktop/sem_7/project_phase_1/dataset/multimodal-hate/training_data/concatenated_features.xlsx'  # Excel file for concatenated features

# Create the output directory if it doesn't exist
os.makedirs(output_npy_dir, exist_ok=True)

# Load video features from the Excel file
def load_vit16_features(vit16_features_path):
    df_vit = pd.read_excel(vit16_features_path)
    
    # Check if 'File Name' column exists to match video names
    if 'File Name' in df_vit.columns:
        vit16_names = df_vit['File Name'].values
        vit16_features = df_vit.drop(columns=['File Name']).values
    else:
        vit16_names = df_vit.index.values
        vit16_features = df_vit.values
    
    return vit16_names, vit16_features

# Load the text embeddings from the .npy files
def load_text_embeddings(text_embeddings_dir):
    text_embeddings = {}
    
    for filename in os.listdir(text_embeddings_dir):
        if filename.endswith('.npy'):
            # Get the name of the file without the extension
            name = os.path.splitext(filename)[0]
            
            # Load the .npy file and store it in the dictionary
            embedding = np.load(os.path.join(text_embeddings_dir, filename))
            text_embeddings[name] = embedding
    
    return text_embeddings

# Load text embeddings
text_embeddings = load_text_embeddings(text_embeddings_dir)

# Load ViT16 video features
vit16_names, vit16_features = load_vit16_features(vit16_features_path)

# List to hold concatenated features and corresponding file names for the Excel file
concatenated_features_list = []
concatenated_names = []

# Iterate over the ViT16 video features and match with text embeddings
for name, vit16_feature in zip(vit16_names, vit16_features):
    # Find the corresponding text embedding by name
    if name in text_embeddings:
        text_feature = text_embeddings[name]
        
        # Concatenate the text and video features
        concatenated_feature = np.concatenate((text_feature, vit16_feature))
        
        # Save the concatenated feature as a .npy file
        output_npy_path = os.path.join(output_npy_dir, f"{name}_concatenated.npy")
        np.save(output_npy_path, concatenated_feature)
        print(f"Saved concatenated feature for {name} to {output_npy_path}")
        
        # Append the concatenated feature and name for the Excel file
        concatenated_features_list.append(concatenated_feature)
        concatenated_names.append(name)
    else:
        print(f"No matching text embedding found for {name}")

# Create a DataFrame for the concatenated features
df_concatenated = pd.DataFrame(concatenated_features_list, index=concatenated_names)
df_concatenated.index.name = 'File Name'

# Save the concatenated features to an Excel file
df_concatenated.to_excel(output_excel_path)
print(f"Concatenated features saved to Excel file: {output_excel_path}")
