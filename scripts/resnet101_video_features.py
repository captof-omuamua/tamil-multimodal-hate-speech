import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet101_Weights  # Import the weight enum for ResNet-101
from PIL import Image
import numpy as np

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the ResNet-101 model pre-trained on ImageNet using the new 'weights' argument
weights = ResNet101_Weights.IMAGENET1K_V1  # Use the appropriate weights
model = models.resnet101(weights=weights)
model.eval()  # Set the model to evaluation mode
model = model.to(device)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),           # Convert image to tensor
    transforms.Normalize(            # Normalize with ImageNet stats
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])

# Directory paths
framewise_feature_dir = '/home/raptor/Desktop/sem_7/project_phase_1/dataset/multimodal-hate/test_data/resnet_101_test/resnet101_framewise_features_test'
avg_video_feature_dir = '/home/raptor/Desktop/sem_7/project_phase_1/dataset/multimodal-hate/test_data/resnet_101_test/resnet101_diff_types_of_temporal_aggregates_test/resnet101_avg_video_frame_feature_test'
video_frames_directory = '/home/raptor/Desktop/sem_7/project_phase_1/dataset/multimodal-hate/test_data/test_frames'

# Create directories if they don't exist
os.makedirs(framewise_feature_dir, exist_ok=True)
os.makedirs(avg_video_feature_dir, exist_ok=True)

# Function to extract features from a single frame
def extract_frame_features(image_path):
    image = Image.open(image_path).convert('RGB')  # Open and convert image to RGB
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Extract features using ResNet-101
    with torch.no_grad():
        features = model(image_tensor)
    
    return features.cpu().numpy().squeeze()  # Squeeze to remove batch dimension

# Function to extract features for all frames of a video and compute average feature vector
def process_video(video_name, video_frame_folder):
    frame_features_list = []

    # Directory to store framewise features for this video
    video_framewise_feature_dir = os.path.join(framewise_feature_dir, video_name)
    os.makedirs(video_framewise_feature_dir, exist_ok=True)

    # Loop through each frame (image) in the video folder
    for frame_file in os.listdir(video_frame_folder):
        if frame_file.endswith('.jpg') or frame_file.endswith('.png'):  # Check for image file types
            frame_path = os.path.join(video_frame_folder, frame_file)
            print(f"Processing frame: {frame_path}")
            
            # Extract features for this frame
            frame_features = extract_frame_features(frame_path)

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
def process_all_videos(video_frames_directory):
    # Loop through each video folder (each folder represents a video)
    for video_folder in os.listdir(video_frames_directory):
        video_folder_path = os.path.join(video_frames_directory, video_folder)
        
        if os.path.isdir(video_folder_path):  # Check if it's a directory (i.e., a video folder)
            print(f"Processing video: {video_folder}")
            process_video(video_folder, video_folder_path)

# Run the processing for all videos in the specified directory
process_all_videos(video_frames_directory)
