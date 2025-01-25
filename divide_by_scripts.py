import os
from pathlib import Path
from shutil import move


def split_png_files_by_script(source_dir: Path):
    # Define the directories within source_dir
    train_dir = source_dir / "train"
    val_dir = source_dir / "validation"
    test_dir = source_dir / "test"

    # Create the directories if they don't exist
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over all .png files in the source directory
    for png_file in source_dir.glob("*.png"):
        file_name = png_file.stem
        script_num = int(file_name.split('_')[1][6:])  # Extract script number from file name

        # Move files to the respective directories based on script number
        if script_num in [1, 2, 3]:
            move(png_file, train_dir / png_file.name)
        elif script_num == 4:
            move(png_file, val_dir / png_file.name)
        elif script_num == 5:
            move(png_file, test_dir / png_file.name)


if __name__ == "__main__":
    # Usage example
    source_directory = Path("spectrograms/Class0") # Change this to the path of cleaned audio files
    split_png_files_by_script(source_directory)