import os
import csv

data_bandcamp = []
data_common = []
index = 0

for i in range(8):
    data_common.append(open(f"/nfs/data/datasets-wav/index/common_{i}.txt", "w"))
    data_bandcamp.append(open(f"/nfs/data/datasets-wav/index/bandcamp_{i}.txt", "w"))

with open("/nfs/data/datasets-mp3/paired-data-index.csv") as fp:
    reader = csv.reader(fp)
    for line in reader:
        path = line[0]
        if "bandcamp" in path:
            data_bandcamp[index % 8].write(f"{path}\n")
        else:
            data_common[index % 8].write(f"{path}\n")
        index += 1

for file in data_common + data_bandcamp:
    file.close()