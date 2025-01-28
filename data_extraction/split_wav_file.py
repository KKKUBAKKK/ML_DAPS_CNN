import os
from pydub import AudioSegment


def split_wav_file(input_file, output_dir, segment_length=3000):
    """
    Splits a given wav file into multiple segments of specified length (in milliseconds)
    and saves them in the specified output directory.

    Args:
    - input_file (str): Path to the input wav file.
    - output_dir (str): Path to the output directory where segments will be saved.
    - segment_length (int): Length of each segment in milliseconds (default is 3000 ms or 3 seconds).
    """

    # Load the audio file
    audio = AudioSegment.from_wav(input_file)
    file_length = len(audio)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Split and save segments
    for i in range(0, file_length, segment_length):
        # Define the segment
        segment = audio[i:i + segment_length]

        # Check if the segment length is less than the specified segment length
        if len(segment) < segment_length:
            break  # Skip the segment if it's shorter than the specified length

        # Define the segment file name
        base_name = os.path.splitext(os.path.basename(input_file))[0]  # Get the file name without extension
        segment_filename = os.path.join(output_dir, f"{base_name}_{i // segment_length + 1}.wav")

        # Export the segment as a wav file
        segment.export(segment_filename, format="wav")

        print(f"Saved {segment_filename}")


# def split_wav_file(input_file, segment_length=3000):
#     """
#     Splits a given wav file into multiple segments of specified length (in milliseconds)
#     and saves them in a structured output directory.
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
#     # Create output directory based on the input file path
#     relative_path = os.path.relpath(input_file, 'data')
#     parent_folder = os.path.dirname(relative_path)  # Get the folder path after 'data'
#     # Create structured output directory based on input file path
#     base_name = os.path.splitext(os.path.basename(input_file))[0]  # Get the file name without extension
#     # parent_folder = os.path.basename(os.path.dirname(input_file))  # Get the folder name of the input file
#     output_base_dir = os.path.join("data", f"{parent_folder}_segments")
#
#     # Ensure output directory exists
#     os.makedirs(output_base_dir, exist_ok=True)
#
#     # Split and save segments
#     for i in range(0, file_length, segment_length):
#         # Define the segment
#         segment = audio[i:i + segment_length]
#
#         # Check if the segment length is less than the specified segment length
#         if len(segment) < segment_length:
#             break  # Skip the segment if it's shorter than the specified length
#
#         # Define the segment file name
#         segment_filename = os.path.join(output_base_dir, f"{base_name}_{i // segment_length + 1}.wav")
#
#         # Export the segment as a wav file
#         segment.export(segment_filename, format="wav")
#
#         print(f"Saved {segment_filename}")


if __name__ == "__main__":
    split_wav_file("data/class_0_cleared/f2_script1_clean.wav")