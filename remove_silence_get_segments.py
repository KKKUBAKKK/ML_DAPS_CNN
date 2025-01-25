# from remove_silence import remove_silence
# from pathlib import Path
# import os
#
# from remove_silence_from_directory import remove_silence_from_directory
# from split_all_files import split_all_files
#
# def remove_silence_get_segments():
#     input_dir = Path('./data/audio_denoised')
#     output_dir = Path('./data/audio_silence_removed')
#     remove_silence_from_directory(input_dir / 'train', output_dir / 'train')
#     remove_silence_from_directory(input_dir / 'validation', output_dir / 'validation')
#     remove_silence_from_directory(input_dir / 'test', output_dir / 'test')
#
#     input_dir = Path('./data/audio_silence_removed')
#     output_dir = Path('./data/audio_segments')
#
#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(output_dir / 'train', exist_ok=True)
#     os.makedirs(output_dir / 'validation', exist_ok=True)
#     os.makedirs(output_dir / 'test', exist_ok=True)
#
#     split_all_files(input_dir / 'train' / 'class_0', output_dir / 'train' / 'class_0')
#     split_all_files(input_dir / 'train' / 'class_1', output_dir / 'train' / 'class_1')
#
#     split_all_files(input_dir / 'validation' / 'class_0', output_dir / 'validation' / 'class_0')
#     split_all_files(input_dir / 'validation' / 'class_1', output_dir / 'validation' / 'class_1')
#
#     split_all_files(input_dir / 'test' / 'class_0', output_dir / 'test' / 'class_0')
#     split_all_files(input_dir / 'test' / 'class_1', output_dir / 'test' / 'class_1')

from concurrent.futures import ThreadPoolExecutor
from remove_silence import remove_silence
from pathlib import Path
import os

from split_all_files import split_all_files

def process_directory(input_dir, output_dir, process_function):
    os.makedirs(output_dir, exist_ok=True)
    with ThreadPoolExecutor() as executor:
        futures = []
        for subdir in input_dir.iterdir():
            if subdir.is_dir():
                for sd in subdir.iterdir():
                    if sd.is_dir():
                        for file in sd.rglob("*.wav"):
                            futures.append(executor.submit(process_function, file, output_dir / subdir.name / sd.name / file.name))
        for future in futures:
            future.result()

def remove_silence_get_segments():
    input_dir = Path('./data/audio_denoised')
    output_dir = Path('./data/audio_silence_removed')
    process_directory(input_dir, output_dir, remove_silence)

    input_dir = Path('./data/audio_silence_removed')
    output_dir = Path('./data/audio_segments')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir / 'train', exist_ok=True)
    os.makedirs(output_dir / 'validation', exist_ok=True)
    os.makedirs(output_dir / 'test', exist_ok=True)

    with ThreadPoolExecutor() as executor:
        futures = []
        for class_dir in ['train', 'validation', 'test']:
            for class_name in ['class_0', 'class_1']:
                futures.append(executor.submit(split_all_files, input_dir / class_dir / class_name, output_dir / class_dir / class_name))
        for future in futures:
            future.result()