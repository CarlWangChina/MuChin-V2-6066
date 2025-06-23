import os
import pickle
import torch
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

def process_file(file_path):
    data = torch.tensor(torch.load(file_path)[1])
    mean = data.mean(dim=0, keepdim=True)
    print(mean.shape)
    with open(file_path + ".jkb", 'wb') as f:
        torch.save(mean, f)

def main():
    with ProcessPoolExecutor() as p:
        dir_path = '/nfs/music-5-test/jukebox/encode/'
        for dirpath, dirnames, filenames in os.walk(dir_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if file_path.endswith('.pkl'):
                    p.submit(process_file, file_path)

if __name__ == '__main__':
    main()