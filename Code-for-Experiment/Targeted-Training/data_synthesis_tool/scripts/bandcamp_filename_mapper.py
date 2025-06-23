import chardet
import urllib.parse
import hashlib
import os
import csv

class StringEncoder:
    def __init__(self):
        self.string_array = []
        self.hash_set = set()

    def add(self, input_string):
        input_bytes = input_string.encode()
        sha256_hash_ori = hashlib.sha256(input_bytes).hexdigest()
        sha256_hash = sha256_hash_ori
        count = 1
        while sha256_hash in self.hash_set:
            sha256_hash = f"{sha256_hash_ori}_{count}"
            count += 1
        self.hash_set.add(sha256_hash)
        encoded_string = urllib.parse.quote(input_bytes)
        self.string_array.append((input_string, encoded_string, sha256_hash))

    def get_array(self):
        return self.string_array

encoder = StringEncoder()

def process_path_list(file_list_path: str):
    with open(file_list_path, 'r') as f:
        file_list = [line.strip() for line in f]
    for file in file_list:
        name = os.path.splitext(file)[0]
        print(file, name)
        encoder.add(name)

process_path_list("/nfs/data/datasets-wav/index/bandcamp_0.txt")
process_path_list("/nfs/data/datasets-wav/index/bandcamp_1.txt")
process_path_list("/nfs/data/datasets-wav/index/bandcamp_2.txt")
process_path_list("/nfs/data/datasets-wav/index/bandcamp_3.txt")
process_path_list("/nfs/data/datasets-wav/index/bandcamp_4.txt")
process_path_list("/nfs/data/datasets-wav/index/bandcamp_5.txt")
process_path_list("/nfs/data/datasets-wav/index/bandcamp_6.txt")
process_path_list("/nfs/data/datasets-wav/index/bandcamp_7.txt")

with open("bandcamp_mapper.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(("path","url_encode","hash"))
    writer.writerows(encoder.string_array)