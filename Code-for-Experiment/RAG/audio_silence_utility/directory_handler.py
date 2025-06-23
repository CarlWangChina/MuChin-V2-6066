import os

def get_valid_directory():
    while True:
        directory_path = input("Please enter the path to the directory containing the audio files: ")
        if os.path.exists(directory_path) and os.path.isdir(directory_path):
            return directory_path
        else:
            print("Invalid directory. Please enter a valid path.")