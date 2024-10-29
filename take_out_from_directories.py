from pathlib import Path
import shutil

def move_files_to_parent_and_delete_empty_dirs(parent_dir: Path):
    # Iterate through all subdirectories in the parent directory
    for sub_dir in parent_dir.iterdir():
        if sub_dir.is_dir():  # Check if it's a directory
            for file in sub_dir.iterdir():  # Iterate through all files in the subdirectory
                if file.is_file():
                    # Move each file to the parent directory
                    shutil.move(str(file), parent_dir / file.name)
            # Remove the subdirectory if it’s empty
            sub_dir.rmdir()
            print(f"Deleted empty directory: {sub_dir}")
        print("All files moved to parent directory.")

# Usage example
if __name__ == "__main__":
    parent_directory = Path("./daps/data")
    move_files_to_parent_and_delete_empty_dirs(parent_directory)