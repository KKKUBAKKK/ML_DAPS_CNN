import os
import shutil
import random

# Ustawienia ścieżek
base_dir = "spectrograms"
output_dir = "spectrograms"
categories = ["class_0", "class_1"]
splits = {"train": 0.6, "validation": 0.2, "test": 0.2}

# Funkcja do tworzenia struktury folderów
def create_directory_structure():
    for split in splits:
        for category in categories:
            os.makedirs(os.path.join(output_dir, split, category), exist_ok=True)

# Funkcja do podziału i przenoszenia plików
def split_and_move_files():
    for category in categories:
        # Lista wszystkich plików .png w danej kategorii
        files = [f for f in os.listdir(os.path.join(base_dir, category)) if f.endswith(".png")]
        random.shuffle(files)  # Losowa kolejność plików

        # Ustalanie ilości plików dla każdego splitu
        num_files = len(files)
        train_split = int(splits["train"] * num_files)
        validation_split = int(splits["validation"] * num_files)

        # Podział plików na zestawy train, validation, test
        train_files = files[:train_split]
        validation_files = files[train_split:train_split + validation_split]
        test_files = files[train_split + validation_split:]

        # Przenoszenie plików
        move_files(train_files, category, "train")
        move_files(validation_files, category, "validation")
        move_files(test_files, category, "test")

# Funkcja pomocnicza do przenoszenia plików
def move_files(files, category, split):
    for file_name in files:
        src_path = os.path.join(base_dir, category, file_name)
        dest_path = os.path.join(output_dir, split, category, file_name)
        shutil.move(src_path, dest_path)

# Tworzenie struktury folderów i przenoszenie plików
create_directory_structure()
split_and_move_files()

print("Pliki zostały pomyślnie przeniesione.")
