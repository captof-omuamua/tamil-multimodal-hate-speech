import os
import torch
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import numpy as np

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained Vision Transformer model and image processor
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
model.eval()  # Set the model to evaluation mode
model = model.to(device)  # Move model to GPU if available

# Image processor for pre-processing (resizing, normalizing)
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

# Directory paths
framewise_feature_dir = '/home/raptor/Desktop/sem_7/project_phase_1/dataset/multimodal-hate/test_data/vit_test/frame_wise_features'
avg_video_feature_dir = '/home/raptor/Desktop/sem_7/project_phase_1/dataset/multimodal-hate/test_data/vit_test/avg_frame_wise_features'
video_frames_directory = '/home/raptor/Desktop/sem_7/project_phase_1/dataset/multimodal-hate/test_data/test_frames'

# Create directories if they don't exist
os.makedirs(framewise_feature_dir, exist_ok=True)
os.makedirs(avg_video_feature_dir, exist_ok=True)

# Function to extract features from a single frame using Vision Transformer
def extract_frame_features_vit(image_path):
    # Open the image and convert it to RGB
    image = Image.open(image_path).convert('RGB')

    # Preprocess the image using the ViT processor
    inputs = processor(images=image, return_tensors="pt").to(device)  # Convert to PyTorch tensor and move to device
    
    # Extract features using Vision Transformer
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the last hidden state (features) and average over the patch dimension
    features = outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()

    return features

# Function to extract features for all frames of a video and compute average feature vector
def process_video_vit(video_name, video_frame_folder):
    frame_features_list = []

    # Directory to store framewise features for this video
    video_framewise_feature_dir = os.path.join(framewise_feature_dir, video_name)
    os.makedirs(video_framewise_feature_dir, exist_ok=True)

    print(f"Processing video: {video_name} in folder: {video_frame_folder}")

    # Loop through each frame (image) in the video folder
    for frame_file in os.listdir(video_frame_folder):
        if frame_file.endswith('.jpg') or frame_file.endswith('.png'):  # Check for image file types
            frame_path = os.path.join(video_frame_folder, frame_file)
            print(f"Processing frame: {frame_path}")

            # Extract features for this frame using Vision Transformer
            frame_features = extract_frame_features_vit(frame_path)

            # Save frame features to a .npy file
            frame_feature_path = os.path.join(video_framewise_feature_dir, f"{os.path.splitext(frame_file)[0]}_features.npy")
            np.save(frame_feature_path, frame_features)

            # Add the frame features to the list
            frame_features_list.append(frame_features)

    # Compute the average feature vector for the video
    if frame_features_list:
        avg_feature_vector = np.mean(np.stack(frame_features_list), axis=0)
        
        # Save the average feature vector for this video
        avg_feature_path = os.path.join(avg_video_feature_dir, f"{video_name}_avg_features.npy")
        np.save(avg_feature_path, avg_feature_vector)

        print(f"Average feature vector for video '{video_name}' saved to {avg_feature_path}")
    else:
        print(f"No frames found for video '{video_name}'")

# Main function to process all videos in the dataset
def process_all_videos_vit(video_frames_directory):
    # Check if the main video frames directory exists
    if not os.path.exists(video_frames_directory):
        print(f"Video frames directory '{video_frames_directory}' does not exist!")
        return
    
    # Loop through each video folder (each folder represents a video)
    video_folders = os.listdir(video_frames_directory)
    if not video_folders:
        print(f"No video folders found in directory '{video_frames_directory}'!")
        return

    for video_folder in video_folders:
        video_folder_path = os.path.join(video_frames_directory, video_folder)

        if os.path.isdir(video_folder_path):  # Check if it's a directory (i.e., a video folder)
            print(f"Found video folder: {video_folder}")
            process_video_vit(video_folder, video_folder_path)
        else:
            print(f"Skipping '{video_folder}' as it is not a folder.")

# Run the processing for all videos in the specified directory
process_all_videos_vit(video_frames_directory)
