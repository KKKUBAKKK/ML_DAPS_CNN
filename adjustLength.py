import librosa
import soundfile as sf
from pathlib import Path

# Define the length of each segment in seconds
SEGMENT_DURATION = 10  # 10 seconds


def extract_segments_from_audio(signal, sample_rate):
    # Load the entire audio file

    # Calculate the number of samples for 10 seconds
    segment_length = SEGMENT_DURATION * sample_rate

    # Extract 3 segments: from the beginning, middle, and end
    segments = {
        "beginning": signal[:segment_length],  # First 10 seconds
        "middle": signal[len(signal) // 2 - segment_length // 2: len(signal) // 2 + segment_length // 2],
        # Middle 10 seconds
        "end": signal[-segment_length:]  # Last 10 seconds
    }

    return segments
