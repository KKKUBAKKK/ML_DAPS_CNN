import os
import shutil


def organize_images(input_directory):
    # Define the person codes for Class1
    class1_codes = ['f1', 'f7', 'f8', 'm3', 'm6', 'm8']

    # Create Class1 and Class0 directories if they don't exist
    class1_directory = os.path.join(input_directory, 'class_1')
    class0_directory = os.path.join(input_directory, 'class_0')

    os.makedirs(class1_directory, exist_ok=True)
    os.makedirs(class0_directory, exist_ok=True)

    # Loop through each file in the input directory
    i = 1
    for filename in os.listdir(input_directory):
        if filename.endswith('.wav'):
            # Extract the person code from the filename
            person_code = filename.split('_')[0]  # Get the first part of the filename

            # Determine the target directory
            if person_code in class1_codes:
                target_directory = class1_directory
                print(f"Moving {filename} to Class1")
            else:
                target_directory = class0_directory
                print(f"Moving {filename} to Class0")

            # Move the file to the appropriate directory
            shutil.move(os.path.join(input_directory, filename), os.path.join(target_directory, filename))
            print(f"Moved {i} files")
            i += 1

if __name__ == "__main__":
    organize_images("./daps/data/train")
    organize_images("./daps/data/validation")
    organize_images("./daps/data/test")