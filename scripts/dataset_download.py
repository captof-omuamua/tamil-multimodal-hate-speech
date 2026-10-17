from datasets import load_dataset
import pandas as pd
import os

# Specify the directory where you want to store the dataset
cache_dir = "/home/raptor/Desktop/sem_7/project_phase_1/dataset/multimodal-hate/training_data/synthetic_data"

# Create the directory if it doesn't exist
os.makedirs(cache_dir, exist_ok=True)

# Load each split of the dataset from Hugging Face and specify the cache directory
try:
    train_dataset = load_dataset("krishan-CSE/Hate_Speech_Tamil", split="train", cache_dir=cache_dir)
    test_dataset = load_dataset("krishan-CSE/Hate_Speech_Tamil", split="test", cache_dir=cache_dir)

    # Convert the 'train' and 'test' splits to Pandas DataFrames
    train_df = pd.DataFrame(train_dataset)
    test_df = pd.DataFrame(test_dataset)

    print("Dataset loaded and stored in specified directory.")
except Exception as e:
    print(f"An error occurred: {e}")
