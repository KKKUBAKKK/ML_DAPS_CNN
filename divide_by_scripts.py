import os
from pathlib import Path
from shutil import move


def split_wav_files_by_script(source_dir: Path):
    # Define the directories within source_dir
    train_dir = source_dir / "train"
    val_dir = source_dir / "validation"
    test_dir = source_dir / "test"

    # Create the directories if they don't exist
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over all .wav files in the source directory
    for wav_file in source_dir.glob("*.png"):
        file_name = wav_file.stem
        script_num = int(file_name.split('_')[1][6:])  # Extract script number from file name

        # Move files to the respective directories based on script number
        if script_num in [1, 2, 3]:
            move(wav_file, train_dir / wav_file.name)
        elif script_num == 4:
            move(wav_file, val_dir / wav_file.name)
        elif script_num == 5:
            move(wav_file, test_dir / wav_file.name)


if __name__ == "__main__":
    # Usage example
    source_directory = Path("path/to/clean/audio") # Change this to the path of cleaned audio files
    split_wav_files_by_script(source_directory)