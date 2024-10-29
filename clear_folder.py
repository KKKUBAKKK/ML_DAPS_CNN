import os
from remove_silence import remove_silence
import argparse

def remove_silence_from_folder(input_folder, min_silence_len=500, silence_thresh=-40):
    # Create output folder if it doesn't exist
    output_folder = input_folder + "_cleared"
    os.makedirs(output_folder, exist_ok=True)

    # Process each .wav file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            remove_silence(input_path, output_path)

    print(f"All files processed. Cleared files saved in: {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove silence from an audio files in whole folder.")
    parser.add_argument("input_path", type=str, help="Path to the input WAV files folder.")

    args = parser.parse_args()

    remove_silence_from_folder(args.input_path)