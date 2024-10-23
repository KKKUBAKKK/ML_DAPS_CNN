import os
import shutil
import random
from pathlib import Path

# Paths to the dataset
dataset_path = Path("./daps/data")  # Replace with actual dataset path
output_base_path = Path("split_data")  # Output directory for splits
output_base_path.mkdir(exist_ok=True)

# Create directories for train, validation, and test
splits = ["train", "validation", "test"]
for split in splits:
    (output_base_path / split).mkdir(exist_ok=True)

# List all environment/device folders
all_folders = [f for f in dataset_path.iterdir() if f.is_dir()]

# Shuffle the folders to ensure randomness
random.shuffle(all_folders)

# Define split ratios
train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15

# Split the data
train_count = int(len(all_folders) * train_ratio)
validation_count = int(len(all_folders) * validation_ratio)

train_folders = all_folders[:train_count]
validation_folders = all_folders[train_count:train_count + validation_count]
test_folders = all_folders[train_count + validation_count:]

# Function to copy or symlink data
def copy_folders(folders, split_name):
    for folder in folders:
        target_path = output_base_path / split_name / folder.name
        shutil.copytree(folder, target_path)  # Use copytree to copy the whole folder
        print(f"Copied {folder} to {target_path}")

# Copy the data to corresponding splits
copy_folders(train_folders, "train")
copy_folders(validation_folders, "validation")
copy_folders(test_folders, "test")

print(f"Data split into train, validation, and test sets successfully!")
