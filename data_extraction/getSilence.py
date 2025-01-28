from pydub import AudioSegment
import webrtcvad
import numpy as np

def remove_voice(audio_file_path, output_file_path, aggressiveness=3):
    # Load audio and convert to mono if needed
    audio = AudioSegment.from_wav(audio_file_path).set_channels(1)

    # Resample audio if necessary
    if audio.frame_rate not in [8000, 16000, 32000, 48000]:
        print(f"Resampling from {audio.frame_rate} Hz to 16000 Hz.")
        audio = audio.set_frame_rate(16000)

    sample_rate = audio.frame_rate

    # Initialize VAD with aggressiveness level (0-3)
    vad = webrtcvad.Vad(aggressiveness)

    # Split audio into 20ms chunks
    frame_duration_ms = 20
    frame_length = int(sample_rate * (frame_duration_ms / 1000))
    audio_data = np.array(audio.get_array_of_samples())

    # Ensure audio data length is a multiple of frame_length
    audio_data = audio_data[:len(audio_data) - (len(audio_data) % frame_length)]

    non_voice_audio = np.array([], dtype=np.int16)

    # Process each frame
    for start in range(0, len(audio_data), frame_length):
        frame = audio_data[start:start + frame_length].tobytes()

        # Detect voice in the frame
        if not vad.is_speech(frame, sample_rate):
            # Append non-voice frames to output
            non_voice_audio = np.concatenate((non_voice_audio, audio_data[start:start + frame_length]))

    # Save the resulting audio without voice
    output_audio = AudioSegment(
        data=non_voice_audio.tobytes(),
        sample_width=audio.sample_width,
        frame_rate=sample_rate,
        channels=1
    )
    output_audio.export(output_file_path, format="wav")
    print(f"Processed audio saved to {output_file_path}")


# Example usage
remove_voice("./daps/data/clean/f1_script1_clean.wav", "./output.wav")