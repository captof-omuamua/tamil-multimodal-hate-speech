import os
import numpy as np
import pandas as pd

# Directory where the average feature vectors are stored
avg_video_feature_dir = '/home/raptor/Desktop/sem_7/project_phase_1/dataset/multimodal-hate/training_data/resnet101_train/resnet101_diff_types_of_temporal_aggregates_train/resnet101_avg_video_frame_feature_train'
output_excel_path = 'project_phase_1/dataset/multimodal-hate/training_data/resnet101_train/avg_video_features_resnet101.xlsx'

# List to store the average feature vectors and their corresponding video names
avg_features_list = []
video_names = []

# Loop through all average feature files in the directory
for avg_feature_file in os.listdir(avg_video_feature_dir):
    if avg_feature_file.endswith('_avg_features.npy'):
        # Extract video name from the filename
        video_name = avg_feature_file.replace('_avg_features.npy', '')
        video_names.append(video_name)
        
        # Load the average feature vector
        avg_feature_path = os.path.join(avg_video_feature_dir, avg_feature_file)
        avg_feature_vector = np.load(avg_feature_path)
        
        # Append the feature vector to the list
        avg_features_list.append(avg_feature_vector)

# Convert the list of average feature vectors into a pandas DataFrame
df_avg_features = pd.DataFrame(avg_features_list, index=video_names)

# Create column headers: 'File Name', 'Embedding_1', 'Embedding_2', ..., etc.
num_features = df_avg_features.shape[1]
headers = ['File Name'] + [f'Embedding_{i+1}' for i in range(num_features)]

# Add the 'File Name' as the first column
df_avg_features.insert(0, 'File Name', video_names)

# Save the DataFrame to an Excel file with the specified headers
df_avg_features.columns = headers
df_avg_features.to_excel(output_excel_path, index=False)

print(f"Excel file with headers saved at: {output_excel_path}")
