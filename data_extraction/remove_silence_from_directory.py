import os
from data_extraction.remove_silence import remove_silence


def remove_silence_from_directory(input_folder, output_folder, min_silence_len=500, silence_thresh=-40):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for dir in os.listdir(input_folder):
        if os.path.isdir(os.path.join(input_folder, dir)):
            for filename in os.listdir(os.path.join(input_folder, dir)):
                if filename.endswith(".wav"):
                    input_path = os.path.join(input_folder, filename)
                    output_path = os.path.join(output_folder / dir, filename)
                    remove_silence(input_path, output_path)

    print(f"All files processed. Cleared files saved in: {output_folder}")

if __name__ == "__main__":
    remove_silence_from_directory('data/class_0')
    remove_silence_from_directory('data/class_1')