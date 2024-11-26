import os
from pathlib import Path
from shutil import move


def divide_by_scripts(dest_dir, source_dir):
    # Define the directories within source_dir
    train_dir = dest_dir / "train"
    val_dir = dest_dir / "validation"
    test_dir = dest_dir / "test"

    # Iterate over all .wav files in the source directory
    for wav_file in source_dir.glob("*.wav"):
        file_name = wav_file.stem
        script_num = int(file_name.split('_')[1][6:])  # Extract script number from file name

        # Move files to the respective directories based on script number
        if script_num in [1, 2, 3]:
            move(wav_file, train_dir / wav_file.name)
        elif script_num == 4:
            move(wav_file, val_dir / wav_file.name)
        elif script_num == 5:
            move(wav_file, test_dir / wav_file.name)

    source_dir.rmdir()

if __name__ == "__main__":
    # Usage example
    source_directory = Path("./daps/data") # Change this to the path of cleaned audio files
    divide_by_scripts(source_directory)