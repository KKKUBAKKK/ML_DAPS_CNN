from random import sample

from create_spectrograms import plot_spectrogram_and_save, IMG_OUTPUT_PATH
from adjustLength import extract_segments_from_audio
import soundfile as sf
from pathlib import Path

TRAIN_IMG_OUTPUT_PATH = Path("./data/spectrograms/train")
VALIDATION_IMG_OUTPUT_PATH = Path("./data/spectrograms/validation")
TEST_IMG_OUTPUT_PATH = Path("./data/spectrograms/test")

TRAIN_AUDIO_PATH = Path("./data/audio/train")
VALIDATION_AUDIO_PATH = Path("./data/audio/validation")
TEST_AUDIO_PATH = Path("./data/audio/test")

def save_spectrograms(dataset_path: Path, output_path: Path):
    audio_files = list(dataset_path.rglob("*.wav"))  # Adjust if your files have a different extension
    print(f"Found {len(audio_files)} files in {dataset_path}")
    print(f"Output path: {output_path.resolve()}")  # Print the resolved output path

    output_path.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists

    j = 0
    for audio_file in audio_files:
        signal, sample_rate = sf.read(audio_file)
        print(f"Processing {audio_file} with sample rate: {sample_rate}")

        # Now, you can pass this signal to your spectrogram function
        output_image_path = output_path / f"{audio_file.stem}.png"

        plot_spectrogram_and_save(signal, sample_rate, output_image_path)
        print(f"Saved spectrogram to {output_image_path}")
        j += 1
        print(f"Processed {j} files")
        # break; # For tests

    print(f"Processed {len(audio_files)} files, saved to {output_path}")


if __name__ == "__main__":
    # process_daps_dataset(TRAIN_AUDIO_PATH / "class_0_cleared_segments", TRAIN_IMG_OUTPUT_PATH / "class_0")
    # process_daps_dataset(TRAIN_AUDIO_PATH / "class_1_cleared_segments", TRAIN_IMG_OUTPUT_PATH / "class_1")

    save_spectrograms(VALIDATION_AUDIO_PATH / "class_0_cleared_segments", VALIDATION_IMG_OUTPUT_PATH / "class_0")
    save_spectrograms(VALIDATION_AUDIO_PATH / "class_1_cleared_segments", VALIDATION_IMG_OUTPUT_PATH / "class_1")

    # process_daps_dataset(TEST_AUDIO_PATH / "class_0_cleared_segments", TEST_IMG_OUTPUT_PATH / "class_0")
    # process_daps_dataset(TEST_AUDIO_PATH / "class_1_cleared_segments", TEST_IMG_OUTPUT_PATH / "class_1")