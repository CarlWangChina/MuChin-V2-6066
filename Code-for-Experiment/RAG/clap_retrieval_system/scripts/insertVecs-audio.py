import os
from pymilvus import connections, db, Collection, CollectionSchema, FieldSchema, DataType, utility
import torch

def traverse_folders(root_folder):
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith(".pt"):
                yield file_path

def create_collect(collection_name, entities):
    if collection_name in utility.list_collections():
        utility.drop_collection(collection_name)
        print("Collection", collection_name, "already exists. Deleting it...")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=512)
    ]
    schema = CollectionSchema(fields)
    c = Collection(collection_name, schema)
    id = 0
    for it in entities:
        c.insert([
            [id],
            [it["name"]],
            [it["embedding"].tolist()]
        ])
        id += 1
    print("insert", collection_name, "num:", id)
    c.flush()
    c.create_index("embeddings", {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    })

entities_full = []
entities_4 = []
entities_6 = []
entities_8 = []
entities_10 = []
entities_12 = []

print("connect milvus")
connections.connect(
    alias="default",
    uri="http://localhost:19530",
    token="root:Milvus",
)

print("create database")
db.using_database("music6w2")

print("existed collections:", utility.list_collections())
print("load datas")

for oripath in traverse_folders('/export/data/clap/encode/audio/cb/'):
    data = torch.load(oripath)
    path = oripath.split('/')[-1]
    entities_full.append({"name": path, "embedding": data["full"].view(-1)})
    entities_4.append({"name": path, "embedding": data["4"]["mean"].view(-1)})
    entities_6.append({"name": path, "embedding": data["6"]["mean"].view(-1)})
    entities_8.append({"name": path, "embedding": data["8"]["mean"].view(-1)})
    entities_10.append({"name": path, "embedding": data["10"]["mean"].view(-1)})
    entities_12.append({"name": path, "embedding": data["12"]["mean"].view(-1)})

print("create collections")
create_collect("cb_collect_full", entities_full)
create_collect("cb_collect_4", entities_4)
create_collect("cb_collect_6", entities_6)
create_collect("cb_collect_8", entities_8)
create_collect("cb_collect_10", entities_10)
create_collect("cb_collect_12", entities_12)