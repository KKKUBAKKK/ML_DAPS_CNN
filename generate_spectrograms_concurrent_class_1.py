import os
import concurrent.futures
from generate_spectrogram_ernest import generate_spectrogram_ernest

def process_folder(folder_path, output_base_dir):
    """
    Processes each .wav file in the given folder by generating a spectrogram
    and saving it to the corresponding output directory.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            audio_path = os.path.join(folder_path, filename)
            base_name = os.path.basename(folder_path) + "_" + os.path.splitext(filename)[0]
            output_path = os.path.join(output_base_dir, f"{base_name}.png")
            generate_spectrogram_ernest(audio_path, output_path)


def generate_spectrograms_for_all_folders(base_input_folder, base_output_folder):
    """
    Generates spectrograms for all .wav files in all subfolders of the specified input folder.
    Each folder's spectrogram generation runs on a separate thread.
    """
    os.makedirs(base_output_folder, exist_ok=True)

    # List all folders in the base input directory
    folder_paths = [
        os.path.join(base_input_folder, folder)
        for folder in os.listdir(base_input_folder)
        if os.path.isdir(os.path.join(base_input_folder, folder))
    ]

    # Use ThreadPoolExecutor to process each folder concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_folder, folder_path, base_output_folder)
            for folder_path in folder_paths
        ]

        # Wait for all threads to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Generated an exception: {exc}")


# Example usage
if __name__ == "__main__":
    generate_spectrograms_for_all_folders("data/class_1_cleared_segments", "spectrograms/Class1")