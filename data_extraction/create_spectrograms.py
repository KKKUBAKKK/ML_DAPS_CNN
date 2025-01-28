from pathlib import Path
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

IMG_OUTPUT_PATH = Path("./test_spectrograms")
SAVE_PARAMS = {"dpi": 300, "bbox_inches": "tight", "transparent": True}

TICKS = np.array([31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000])
TICK_LABELS = np.array(["31.25", "62.5", "125", "250", "500", "1k", "2k", "4k", "8k"])


def plot_spectrogram_and_save(signal, fs, output_path: Path, fft_size=2048, hop_size=None, window_size=None):
    if not window_size:
        window_size = fft_size

    if not hop_size:
        hop_size = window_size // 4

    stft = librosa.stft(
        signal,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=window_size,
        center=False,
    )
    spectrogram = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

    plt.figure(figsize=(10, 4))
    img = librosa.display.specshow(
        spectrogram_db,
        y_axis="log",
        x_axis="time",
        sr=fs,
        hop_length=hop_size,
        cmap="inferno",
    )

    # Remove axes
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, **SAVE_PARAMS)
    plt.close()


def process_daps_dataset(dataset_path: Path):
    audio_files = list(dataset_path.glob("**/*.wav"))  # Adjust if your files have a different extension
    for audio_file in audio_files:
        signal, sample_rate = sf.read(audio_file)
        print(f"Processing {audio_file} with sample rate: {sample_rate}")
        output_image_path = IMG_OUTPUT_PATH / audio_file.stem
        plot_spectrogram_and_save(signal, sample_rate, output_image_path)
        break;


# Just for testing
if __name__ == "__main__":
    dataset_path = Path("./daps/data/clean")  # Replace with the correct path to DAPS dataset
    process_daps_dataset(dataset_path)