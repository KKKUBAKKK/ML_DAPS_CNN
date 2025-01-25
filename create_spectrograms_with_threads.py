from time import process_time_ns

from get_spectrograms import get_spectrograms
from remove_silence_get_segments import remove_silence_get_segments


def create_spectrograms_with_threads():
    remove_silence_get_segments()
    print("All files processed.")
    get_spectrograms()
    print("All files processed.")

if __name__ == "__main__":
    create_spectrograms_with_threads()
    print("All files processed.")