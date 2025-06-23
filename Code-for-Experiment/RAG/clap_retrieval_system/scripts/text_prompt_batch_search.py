import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server"))

import ClapProcessor
import torch
import json
import vec_search

device = "cpu"
table = "clap"
processor_clap = ClapProcessor.ClapProcessor(device=device)
prompts = []
vecs = processor_clap.processText(prompts)
c = vec_search.Collection(table)
c.load()
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10},
}
outputs = []
for i, vec in enumerate(vecs):
    result = c.search(vec, "embeddings", search_params, limit=10, output_fields=["path"])
    res_path = [[row.path for row in item] for item in result]
    outputs.append({
        "prompt": prompts[i],
        "res": res_path
    })

with open("outputs.json", "w") as f:
    json.dump(outputs, f)