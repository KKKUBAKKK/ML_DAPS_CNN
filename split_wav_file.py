# from pydub import AudioSegment
# import os
#
# def split_wav_file(input_file, segment_length=3000):
#     """
#     Splits a given wav file into multiple segments of specified length (in milliseconds).
#
#     Args:
#     - input_file (str): Path to the input wav file.
#     - segment_length (int): Length of each segment in milliseconds (default is 3000 ms or 3 seconds).
#     """
#
#     # Load the audio file
#     audio = AudioSegment.from_wav(input_file)
#     file_length = len(audio)
#
#     # Create output directory based on the input file name
#     base_name = os.path.splitext(os.path.basename(input_file))[0]  # Get the file name without extension
#     output_dir = f"{base_name}_segments"
#
#     # Ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)
#
#     # Split and save segments
#     for i in range(0, file_length, segment_length):
#         # Define the start and end of the segment
#         segment = audio[i:i + segment_length]
#
#         # Define the segment file name
#         segment_filename = os.path.join(output_dir, f"segment_{i // segment_length + 1}.wav")
#
#         # Export the segment as a wav file
#         segment.export(segment_filename, format="wav")
#
#         print(f"Saved {segment_filename}")
#
# if __name__ == "__main__":
#     split_wav_file("data/tmp/f2_script1_clean.wav")

import os
from pydub import AudioSegment


def split_wav_file(input_file, segment_length=3000):
    """
    Splits a given wav file into multiple segments of specified length (in milliseconds)
    and saves them in a structured output directory.

    Args:
    - input_file (str): Path to the input wav file.
    - segment_length (int): Length of each segment in milliseconds (default is 3000 ms or 3 seconds).
    """

    # Load the audio file
    audio = AudioSegment.from_wav(input_file)
    file_length = len(audio)

    # Create structured output directory based on input file path
    base_name = os.path.splitext(os.path.basename(input_file))[0]  # Get the file name without extension
    parent_folder = os.path.basename(os.path.dirname(input_file))  # Get the folder name of the input file
    output_base_dir = os.path.join("data", f"{parent_folder}_segments", f"{base_name}_segments")

    # Ensure output directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    # Split and save segments
    for i in range(0, file_length, segment_length):
        # Define the segment
        segment = audio[i:i + segment_length]

        # Define the segment file name
        segment_filename = os.path.join(output_base_dir, f"segment_{i // segment_length + 1}.wav")

        # Export the segment as a wav file
        segment.export(segment_filename, format="wav")

        print(f"Saved {segment_filename}")


if __name__ == "__main__":
    split_wav_file("data/class_0_cleared/f2_script1_clean.wav")