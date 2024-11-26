import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
plt.switch_backend('Agg')


def generate_spectrogram_ernest(audio_path, output_path):
    y, sr = librosa.load(audio_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=512)
    S_dB = librosa.power_to_db(S, ref=np.max)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.axis('off')
    plt.imshow(S_dB, aspect='auto', origin='lower')  # użycie imshow jako zamiennik specshow
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Spectrogram saved at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate a spectrogram from an audio file.")
    parser.add_argument("audio_path", type=str, help="Path to the input audio file.")
    parser.add_argument("--output_path", type=str, default="temp_spectrograms/temp_spectrogram.png",
                        help="Path to save the output spectrogram image (default: 'temp_spectrograms/temp_spectrogram.png').")

    args = parser.parse_args()
    generate_spectrogram_ernest(args.audio_path, args.output_path)


if __name__ == "__main__":
    main()
