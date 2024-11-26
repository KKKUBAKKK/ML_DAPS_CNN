import os
from pathlib import Path
import shutil
import random

from fontTools.misc.classifyTools import classify

from classify_audio import classify_audio
from divide_by_scripts import divide_by_scripts
from take_out_from_directories import take_out_from_directories


def divide_daps_dataset(dataset_path: Path, output_base_path: Path = Path('data/audio'), script: bool = True, copy: bool = False):
    # Create directories for train, validation, and test
    splits = ["train", "validation", "test"]
    for split in splits:
        (output_base_path / split).mkdir(exist_ok=True)
        (output_base_path / split / "class_0").mkdir(exist_ok=True)
        (output_base_path / split / "class_1").mkdir(exist_ok=True)

    # List all environment/device folders
    all_folders = [f for f in dataset_path.iterdir() if f.is_dir()]

    # Remove cleanraw and produced folders
    all_folders = [f for f in all_folders if f.name not in ["cleanraw", "produced", "sample", "supplementary_files"]]

    # Split data randomly in folders
    if not script:
        print("Splitting data directories randomly in folders...")

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

        print(f"Train folders: {train_folders}")
        print(f"Validation folders: {validation_folders}")
        print(f"Test folders: {test_folders}")

        # Put the data from lists into corresponding directories
        if copy:
            print("Copying data into train, validation, and test sets...")

            copy_directories(train_folders, Path(dataset_path), output_base_path / "train")
            copy_directories(validation_folders,Path(dataset_path) , output_base_path / "validation")
            copy_directories(test_folders,Path(dataset_path) , output_base_path / "test")

            print(f"Data copied into train, validation, and test sets successfully!")
        else:
            print("Moving data into train, validation, and test sets...")

            move_directories(train_folders, Path(dataset_path), output_base_path / "train")
            move_directories(validation_folders, Path(dataset_path), output_base_path / "validation")
            move_directories(test_folders, Path(dataset_path), output_base_path / "test")

            print(f"Data moved into train, validation, and test sets successfully!")

        # Take out files from directories
        take_out_from_directories(output_base_path / "train")
        take_out_from_directories(output_base_path / "validation")
        take_out_from_directories(output_base_path / "test")

        print(f"Data split into train, validation, and test sets successfully!")
    else:
        print("Splitting data directories by scripts...")

        # Copy or move the data
        for folder in all_folders:
            if copy:
                shutil.copytree(dataset_path / folder.name, output_base_path / folder.name)
            else:
                shutil.move(dataset_path / folder.name, output_base_path / folder.name)

        # Take out files from directories
        take_out_from_directories(output_base_path)

        # Split data by scripts
        for folder in all_folders:
            divide_by_scripts(folder)

        print(f"Data split by scripts successfully!")

    # Split into class_0 and class_1
    classify_audio(output_base_path / "train")
    classify_audio(output_base_path / "validation")
    classify_audio(output_base_path / "test")

    print(f"Data classified into class_0 and class_1 successfully!")



# Function to move directories
def move_directories(dirs: list, source_dir: Path, destination_dir: Path):
    # Ensure the destination directory exists
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through all directories in the source directory
    for item in dirs:
        if item.is_dir():
            # Move each directory to the destination directory
            shutil.move(source_dir / item.name, destination_dir / item.name)
            print(f"Moved directory: {item.name} to {destination_dir}")


# Function to copy directories
def copy_directories(dirs: list, source_dir: Path, destination_dir: Path):
    # Ensure the destination directory exists
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through all directories in the source directory
    for item in dirs:
        if item.is_dir():
            # Copy each directory to the destination directory
            shutil.copytree(source_dir / item.name, destination_dir / item.name)
            print(f"Copied directory: {item.name} to {destination_dir}")

# Usage example
if __name__ == "__main__":
    # Usage example
    dataset_path = Path("./daps")  # Replace with actual dataset path
    output_base_path = Path('data/audio')  # Output directory for splits
    divide_daps_dataset(dataset_path, output_base_path, script=True, copy=False)
    # divide_daps_dataset(dataset_path, output_base_path, script=False, copy=False)
    # divide_daps_dataset(dataset_path, output_base_path, script=False, copy=True)
    # divide_daps_dataset(dataset_path, output_base_path, script=True, copy=True)
