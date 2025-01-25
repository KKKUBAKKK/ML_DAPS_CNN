import os
import shutil
import argparse


def organize_spectrograms(source_dir, destination_dir):
    # Zdefiniuj mapowanie skryptów do folderów
    mapping = {
        "1": "train",
        "2": "train",
        "3": "train",
        "4": "test",
        "5": "validation"
    }

    # Iteruj przez klasy (Class0, Class1)
    for class_folder in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_folder)

        if os.path.isdir(class_path):  # Upewnij się, że to folder
            for file_name in os.listdir(class_path):
                # Wyszukaj numer skryptu w nazwie pliku
                for script_num, folder in mapping.items():
                    if f"_script{script_num}_" in file_name:
                        # Ścieżka docelowa
                        target_dir = os.path.join(destination_dir, folder, class_folder)
                        os.makedirs(target_dir, exist_ok=True)  # Utwórz foldery docelowe

                        # Przenieś plik
                        source_file = os.path.join(class_path, file_name)
                        target_file = os.path.join(target_dir, file_name)
                        shutil.move(source_file, target_file)
                        print(f"Moved: {source_file} -> {target_file}")
                        break


if __name__ == "__main__":
    # Obsługa argumentów z terminala
    parser = argparse.ArgumentParser(description="Organize mfcc_spectrograms into train, test, and validation folders.")
    parser.add_argument("source_dir", type=str, help="Source directory containing Class0 and Class1 folders")
    parser.add_argument("destination_dir", type=str, help="Destination directory for organized mfcc_spectrograms")

    args = parser.parse_args()

    # Uruchomienie funkcji
    organize_spectrograms(args.source_dir, args.destination_dir)