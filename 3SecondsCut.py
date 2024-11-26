# here you have to do
# pip install pydub
# before (unless you already have it installed)

from pydub import AudioSegment
import os


def split_wav_file(input_file, segment_length=3000):
    """
    Splits a given wav file into multiple segments of specified length (in milliseconds).

    Args:
    - input_file (str): Path to the input wav file.
    - segment_length (int): Length of each segment in milliseconds (default is 3000 ms or 3 seconds).
    """

    # Load the audio file
    audio = AudioSegment.from_wav(input_file)
    file_length = len(audio)

    # Create output directory based on the input file name
    base_name = os.path.splitext(os.path.basename(input_file))[0]  # Get the file name without extension
    output_dir = f"{base_name}_segments"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Split and save segments
    for i in range(0, file_length, segment_length):
        # Define the start and end of the segment
        segment = audio[i:i + segment_length]

        # Define the segment file name
        segment_filename = os.path.join(output_dir, f"segment_{i // segment_length + 1}.wav")

        # Export the segment as a wav file
        segment.export(segment_filename, format="wav")

        print(f"Saved {segment_filename}")