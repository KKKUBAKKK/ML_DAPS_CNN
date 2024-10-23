from random import sample

from create_spetrograms import plot_spectrogram_and_save, IMG_OUTPUT_PATH
from adjustLength import extract_segments_from_audio
import soundfile as sf
from pathlib import Path

TRAIN_IMG_OUTPUT_PATH = Path("./spectrograms/train")
VALIDATION_IMG_OUTPUT_PATH = Path("./spectrograms/validation")
TEST_IMG_OUTPUT_PATH = Path("./spectrograms/test")

TRAIN_AUDIO_PATH = Path("./split_data/train")
VALIDATION_AUDIO_PATH = Path("./split_data/validation")
TEST_AUDIO_PATH = Path("./split_data/test")

def process_daps_dataset(dataset_path: Path, output_path: Path):
    audio_files = list(dataset_path.rglob("*.wav"))  # Adjust if your files have a different extension
    print(f"Found {len(audio_files)} files in {dataset_path}")
    j = 0
    for audio_file in audio_files:
        signal, sample_rate = sf.read(audio_file)
        duration = len(signal) / sample_rate
        print(f"Processing {audio_file} with sample rate: {sample_rate}")

        # Get three 10s segments
        if duration < 50:
            continue
        segments = extract_segments_from_audio(signal, sample_rate)

        i = 1;
        for signal in segments.values():
            # Now, you can pass this signal to your spectrogram function
            output_image_path = output_path / f"{audio_file.stem}_{i}.png"
            plot_spectrogram_and_save(signal, sample_rate, output_image_path)
            print(f"Saved spectrogram to {output_image_path}")
            i += 1
        j += 1
        print(f"Processed {j} files")
        # break; # For tests

    print(f"Processed {len(audio_files)} files, saved to {output_path}")


if __name__ == "__main__":
    process_daps_dataset(TRAIN_AUDIO_PATH, TRAIN_IMG_OUTPUT_PATH)
    # process_daps_dataset(VALIDATION_AUDIO_PATH, VALIDATION_IMG_OUTPUT_PATH)
    # process_daps_dataset(TEST_AUDIO_PATH, TEST_IMG_OUTPUT_PATH)