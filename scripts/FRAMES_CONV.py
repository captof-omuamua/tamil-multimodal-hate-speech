import cv2
import os

# Function to convert each video into frames and store them in a directory named after the video
def video_to_frames(input_dir, output_dir):
    # Ensure the output directory exists, or create it if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through all the video files in the input directory
    for video_file in os.listdir(input_dir):
        video_path = os.path.join(input_dir, video_file)

        # Check if the file is a video (by extension)
        if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # Create a directory for the current video's frames
            video_name = os.path.splitext(video_file)[0]  # Strip the file extension to get the video name
            video_output_dir = os.path.join(output_dir, video_name)
            
            # If the folder doesn't exist, create it
            if not os.path.exists(video_output_dir):
                os.makedirs(video_output_dir)

            # Capture the video
            video_capture = cv2.VideoCapture(video_path)
            frame_number = 0

            # Read and save each frame from the video
            while True:
                success, frame = video_capture.read()
                if not success:
                    break  # Break the loop when the video ends

                # Save the frame as a .jpg file in the corresponding video folder
                frame_filename = os.path.join(video_output_dir, f"frame_{frame_number:05d}.jpg")
                cv2.imwrite(frame_filename, frame)
                frame_number += 1

            video_capture.release()
            print(f"Frames from '{video_name}' have been stored in '{video_output_dir}'.")

# Example usage
input_directory = "/home/raptor/Desktop/sem_7/project_phase_1/dataset/multimodal-hate/training_data/Video"  # Replace with the path to your videos
output_directory = "/home/raptor/Desktop/sem_7/project_phase_1/dataset/multimodal-hate/training_data/training_frames"  # Replace with the desired output directory

video_to_frames(input_directory, output_directory)
