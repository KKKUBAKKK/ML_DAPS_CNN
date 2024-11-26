import os
import shutil


def classify_audio(source_dir):
    # Define the person codes for Class1
    class1_codes = ['f1', 'f7', 'f8', 'm3', 'm6', 'm8']

    # Create Class1 and Class0 directories if they don't exist
    class1_directory = source_dir / 'class_1'
    class0_directory = source_dir / 'class_0'

    # Loop through each file in the input directory
    for filename in os.listdir(source_dir):
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
            shutil.move(os.path.join(source_dir, filename), os.path.join(target_directory, filename))