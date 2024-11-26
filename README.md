# Project Title

This project uses the DAPS dataset to teach a CNN how to classify audio files.
It classifies the audio files into 2 classes 0 and 1. The aim is that it should
'let in' only the people whose voice is recognized by the model as from class 1.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Downloading the DAPS dataset](#downloading-the-daps-dataset)
  - [Division into sets and classes](#division-into-sets-and-classes)
  - [Noise Removal](#noise-removal)
  - [Silence Removal](#silence-removal)
  - [Division into segments](#division-into-segments)
  - [Spectrogram Generation](#spectrogram-generation)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/KKKUBAKKK/ML_DAPS_CNN.git
    cd ML_DAPS_CNN
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Downloading the DAPS dataset

To download the DAPS dataset, follow these steps:

1. Visit the [DAPS dataset website](https://ccrma.stanford.edu/damp/).
2. Navigate to the download section and download the dataset files. You may need to register or provide some information to access the download links.
3. Once downloaded, extract the files to a directory of your choice. For example, you can extract them to `./data/daps/`.
4. Ensure the extracted files are organized correctly, typically with subdirectories for different audio files.

After downloading and extracting the dataset, you can proceed with the next steps in the project.

### Division into sets and classes

To divide the DAPS dataset into training, validation, and test sets, use the `divide_daps_dataset` 
function from `divide_daps_dataset.py`:

```python
from pathlib import Path
from divide_daps_dataset import divide_daps_dataset

dataset_path = Path("./daps")  # Replace with actual dataset path
output_base_path = Path('data/audio')  # Output directory for split dataset
divide_daps_dataset(dataset_path, output_base_path, script=True, copy=False)
```

### Noise Removal

To remove noise from audio files in a directory, use the `remove_noise_from_directory` function from `remove_noise.py`:

```python
from pathlib import Path
from remove_noise import remove_noise_from_directory

input_dir = Path('./data/audio')
output_dir = Path('./data/audio_denoised')
remove_noise_from_directory(input_dir / 'train', output_dir / 'train')
remove_noise_from_directory(input_dir / 'validation', output_dir / 'validation')
remove_noise_from_directory(input_dir / 'test', output_dir / 'test')
```

### Silence Removal

To remove silence from audio files in a directory, use the `remove_silence_from_directory` function from `remove_silence_from_directory.py`:

```python
from pathlib import Path
from remove_silence_from_directory import remove_silence_from_directory

input_dir = Path('./data/audio_denoised')
output_dir = Path('./data/audio_silence_removed')
remove_silence_from_directory(input_dir / 'train', output_dir / 'train')
remove_silence_from_directory(input_dir / 'validation', output_dir / 'validation')
remove_silence_from_directory(input_dir / 'test', output_dir / 'test')
```

### Division into segments

To divide audio files into segments, use the `split_all_files` function from `ssplit_all_files.py`:

```python
from pathlib import Path    
from split_all_files import split_all_files
import os
    
input_dir = Path('./data/audio_silence_removed')
output_dir = Path('./data/audio_segments')

os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir / 'train', exist_ok=True)
os.makedirs(output_dir / 'validation', exist_ok=True)
os.makedirs(output_dir / 'test', exist_ok=True)

split_all_files(input_dir / 'train' / 'class_0', output_dir / 'train' / 'class_0')
split_all_files(input_dir / 'train' / 'class_1', output_dir / 'train' / 'class_1')

split_all_files(input_dir / 'validation' / 'class_0', output_dir / 'validation' / 'class_0')
split_all_files(input_dir / 'validation' / 'class_1', output_dir / 'validation' / 'class_1')

split_all_files(input_dir / 'test' / 'class_0', output_dir / 'test' / 'class_0')
split_all_files(input_dir / 'test' / 'class_1', output_dir / 'test' / 'class_1')
```

### Spectrogram Generation

To generate spectrograms for audio files, use the `save_spectrograms` function from `save_spectrograms.py`:

```python
from pathlib import Path
from save_spectrograms import save_spectrograms
import os

dataset_path = Path('./data/audio_segments')
output_path = Path('./data/spectrograms')

os.makedirs(output_path, exist_ok=True)
os.makedirs(output_path / 'train', exist_ok=True)
os.makedirs(output_path / 'validation', exist_ok=True)
os.makedirs(output_path / 'test', exist_ok=True)

save_spectrograms(dataset_path / 'train' / 'class_0', output_path / 'train' / 'class_0')
save_spectrograms(dataset_path / 'train' / 'class_1', output_path / 'train' / 'class_1')

save_spectrograms(dataset_path / 'validation' / 'class_0', output_path / 'validation' / 'class_0')
save_spectrograms(dataset_path / 'validation' / 'class_1', output_path / 'validation' / 'class_1')

save_spectrograms(dataset_path / 'test' / 'class_0', output_path / 'test' / 'class_0')
save_spectrograms(dataset_path / 'test' / 'class_1', output_path / 'test' / 'class_1')
```

### Model Training

To train the model, use the `main` function from `main.py`:

```python
from main import main
from pathlib import Path

data_dir = Path("./data/spectrograms")  # Update this to your dataset path
main(True, data_dir, num_epochs=25, batch_size=32, learning_rate=0.001, model_save_path="new_best_daps_cnn.pth")
```

### Model Evaluation

To evaluate the model, use the `main` function from `main.py`, you can just navigate to `main.py` and run the file with your path or use this code:

```python
from main import main
from pathlib import Path

data_dir = Path("./data/spectrograms")  # Update this to your dataset path
main(False, data_dir, num_epochs=25, batch_size=32, learning_rate=0.001, model_save_path="new_best_daps_cnn.pth")
```
