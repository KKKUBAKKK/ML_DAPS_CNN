import os
from pathlib import Path
import shutil
import random

from fontTools.misc.classifyTools import classify

from classify_audio import classify_audio
from divide_by_scripts import divide_by_scripts
from take_out_from_directories import take_out_from_directories


def divide_daps_dataset(dataset_path, output_base_path, copy):
    # Create directories for train, validation, and test
    splits = ["train", "validation", "test"]
    for split in splits:
        (output_base_path / split).mkdir(exist_ok=True)
        (output_base_path / split / "class_0").mkdir(exist_ok=True)
        (output_base_path / split / "class_1").mkdir(exist_ok=True)

    # List all environment/device folders
    all_folders = [f for f in dataset_path.iterdir() if f.is_dir()]
    all_folders = [f for f in all_folders if f.name not in ["cleanraw", "produced", "sample", "supplementary_files"]]

    print("Splitting data directories by scripts...")

    # Copy or move the data
    for folder in all_folders:
        if copy:
            shutil.copytree(dataset_path / folder.name, output_base_path / folder.name)
        else:
            shutil.move(dataset_path / folder.name, output_base_path / folder.name)

    all_folders = [f for f in output_base_path.iterdir() if f.is_dir()]
    all_folders = [f for f in all_folders if f.name not in ["test", "train", "validation"]]

    for folder in all_folders:
        divide_by_scripts(output_base_path, folder)

    classify_audio(output_base_path / "train")
    classify_audio(output_base_path / "validation")
    classify_audio(output_base_path / "test")

# Usage example
if __name__ == "__main__":
    # Usage example
    dataset_path = Path("daps")  # Replace with actual dataset path
    output_base_path = Path('data/audio')  # Output directory for splits
    divide_daps_dataset(dataset_path, output_base_path, copy=True)

