import os
import subprocess
import shutil
from concurrent.futures import ProcessPoolExecutor

src_dir = '/nfs/data/datasets-mp3/dyqy/'
dest_dir = '/nfs/data/datasets-mp3/dyqy-fix/'

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

def convert_mp3(source_file, dest_dir):
    basename = os.path.basename(source_file)
    destination_file = os.path.join(dest_dir, basename)

    command = ['ffmpeg', '-i', source_file, '-y', destination_file]
    try:
        subprocess.run(command, check=True)
        print(f"Converted {source_file} to {destination_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {source_file}: {e}")

def process_mp3_files(src_dir, dest_dir):
    mp3_files = [os.path.join(src_dir, filename) for filename in os.listdir(src_dir) if filename.lower().endswith('.mp3')]

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(convert_mp3, source_file, dest_dir) for source_file in mp3_files]

        for future in futures:
            future.result()

process_mp3_files(src_dir, dest_dir)