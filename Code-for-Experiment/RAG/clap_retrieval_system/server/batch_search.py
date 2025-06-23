import hashlib
from bottle import Bottle, route, run, request, response, static_file
import vec_search
import json
import csv

data = []
with open('clap_key_limit_12s.csv', 'r', newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)
    for row in csv_reader:
        res, prompt = vec_search.search_data_by_audio([row[2]], table="cb_collect_full", translate=False)
        res_songid = [[r.path.split("_")[0], r.distance] for r in res[0]]
        output = {"search": row, "result": res_songid[0:int(row[3])], "prompt": prompt}
        print(row)
        data.append(output)

with open("batch_search.json", 'w') as json_file:
    json.dump(data, json_file, indent=4)