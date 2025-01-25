# from pathlib import Path
# from save_spectrograms import save_spectrograms
# import os
#
# def get_spectrograms():
#     dataset_path = Path('./data/audio_segments')
#     output_path = Path('./data/spectrograms')
#
#     os.makedirs(output_path, exist_ok=True)
#     os.makedirs(output_path / 'train', exist_ok=True)
#     os.makedirs(output_path / 'validation', exist_ok=True)
#     os.makedirs(output_path / 'test', exist_ok=True)
#
#     save_spectrograms(dataset_path / 'train' / 'class_0', output_path / 'train' / 'class_0')
#     save_spectrograms(dataset_path / 'train' / 'class_1', output_path / 'train' / 'class_1')
#
#     save_spectrograms(dataset_path / 'validation' / 'class_0', output_path / 'validation' / 'class_0')
#     save_spectrograms(dataset_path / 'validation' / 'class_1', output_path / 'validation' / 'class_1')
#
#     save_spectrograms(dataset_path / 'test' / 'class_0', output_path / 'test' / 'class_0')
#     save_spectrograms(dataset_path / 'test' / 'class_1', output_path / 'test' / 'class_1')

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
from save_spectrograms import save_spectrograms

def get_spectrograms():
    dataset_path = Path('./data/audio_segments')
    output_path = Path('./data/spectrograms')

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path / 'train', exist_ok=True)
    os.makedirs(output_path / 'validation', exist_ok=True)
    os.makedirs(output_path / 'test', exist_ok=True)

    def process_class(class_dir, class_name):
        save_spectrograms(dataset_path / class_dir / class_name, output_path / class_dir / class_name)

    with ThreadPoolExecutor() as executor:
        futures = []
        for class_dir in ['test']:
            for class_name in ['class_0']:
                futures.append(executor.submit(process_class, class_dir, class_name))
        for future in futures:
            future.result()

if __name__ == "__main__":
    get_spectrograms()
    print("All files processed.")