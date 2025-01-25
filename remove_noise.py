import threading

import soundfile as sf
import librosa
from pathlib import Path
import noisereduce as nr

from create_spectrograms import plot_spectrogram_and_save


def remove_noise_from_directory(input_dir, output_dir, noise_clip_length=1.0,
                                noise_reduction_type='stationary',
                                prop_decrease=1.0):
    # Iterate through all files in the input directory
    for audio_file in input_dir.rglob("*.wav"):
        # Remove noise from each audio file
        remove_noise_from_wav(audio_file, output_dir / audio_file.name, noise_clip_length,
                              noise_reduction_type, prop_decrease)


def remove_noise_from_wav(input_file, output_file=None,
                          noise_clip_length=1.0,  # seconds of noise to sample
                          noise_reduction_type='stationary',
                          prop_decrease=1.0):  # Amount of noise to reduce
    """
    Advanced noise reduction for WAV audio files using multiple techniques.

    Parameters:
    -----------
    input_file : str
        Path to the input .wav audio file
    output_file : str, optional
        Path to save the noise-reduced audio file.
        If None, appends '_denoised' to the input filename.
    noise_clip_length : float, optional (default=1.0)
        Length of noise clip to sample from the beginning of the audio (in seconds)
    noise_reduction_type : str, optional (default='stationary')
        Type of noise reduction: 'stationary' or 'nonstationary'
    prop_decrease : float, optional (default=1.0)
        Proportion of noise to reduce (0-1 range)
    n_std_thresh_stationary : float, optional (default=1.5)
        Threshold for stationary noise reduction
    n_std_thresh_nonstationary : float, optional (default=1.5)
        Threshold for non-stationary noise reduction

    Returns:
    --------
    numpy.ndarray
        Noise-reduced audio time series
    """
    # Load the audio file with original sample rate
    audio, sample_rate = sf.read(input_file)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Estimate noise profile from the beginning of the audio
    noise_samples = int(noise_clip_length * sample_rate)
    noise_clip = audio[:noise_samples]

    # Perform noise reduction
    try:
        if noise_reduction_type == 'stationary':
            # Stationary noise reduction (works well for consistent background noise)
            reduced_audio = nr.reduce_noise(
                y=audio,
                y_noise=noise_clip,
                sr=sample_rate,
                prop_decrease=prop_decrease,
            )
        else:
            # Non-stationary noise reduction (for varying background noise)
            reduced_audio = nr.reduce_noise(
                y=audio,
                y_noise=noise_clip,
                sr=sample_rate,
                prop_decrease=prop_decrease,
                stationary=False
            )
    except Exception as e:
        print(f"Noise reduction failed: {e}")
        return audio

    # Normalize the audio to prevent clipping
    reduced_audio = librosa.util.normalize(reduced_audio)

    # Determine output filename
    if output_file is None:
        output_file = input_file.rsplit('.', 1)[0] + '_denoised.wav'
    # elif output_file.suffix == '.wav':
    #     output_file = output_file.with_stem(output_file.stem + '_denoised')

    # Save the processed audio
    sf.write(output_file, reduced_audio, sample_rate)

    print(f"Saved denoised audio to: {output_file}")

    return reduced_audio


# Example usage for single audio file
if __name__ == "__main__":
    input_dir = Path('./data/audio')
    output_dir = Path('./data/audio_denoised')
    # train0thread = threading.Thread(target=remove_noise_from_directory(input_dir / 'train' / 'class_0', output_dir / 'train' / 'class_0'))
    train1thread = threading.Thread(target=remove_noise_from_directory(input_dir / 'train' / 'class_1', output_dir / 'train' / 'class_1'))
    validate0thread = threading.Thread(target=remove_noise_from_directory(input_dir / 'validation' / 'class_0', output_dir / 'validation' / 'class_0'))
    validate1thread = threading.Thread(target=remove_noise_from_directory(input_dir / 'validation' / 'class_1', output_dir / 'validation' / 'class_1'))
    test0thread = threading.Thread(target=remove_noise_from_directory(input_dir / 'test' / 'class_0', output_dir / 'test' / 'class_0'))
    test1thread = threading.Thread(target=remove_noise_from_directory(input_dir / 'test' / 'class_1', output_dir / 'test' / 'class_1'))

    # train0thread.start()
    train1thread.start()
    validate0thread.start()
    validate1thread.start()
    test0thread.start()
    test1thread.start()

    # train0thread.join()
    train1thread.join()
    print("Train done")
    validate0thread.join()
    validate1thread.join()
    print("Validation done")
    test0thread.join()
    test1thread.join()
    print("Test done")