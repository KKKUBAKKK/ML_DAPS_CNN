import os
from split_wav_file import split_wav_file

def split_all_files(input_folder, segment_length=3000):
    """
    Splits all .wav files in a specified folder into multiple segments of specified length,
    saving the results in a new folder named input_folder + '_splitted'.

    Args:
    - input_folder (str): Path to the folder containing .wav files.
    - segment_length (int): Length of each segment in milliseconds (default is 3000 ms or 3 seconds).
    """

    # Process each .wav file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            input_file = os.path.join(input_folder, filename)
            print(f"Processing file: {input_file}")
            split_wav_file(input_file, segment_length)

    print(f"All files processed.")


if __name__ == "__main__":
    split_all_files("data/class_1_cleared")