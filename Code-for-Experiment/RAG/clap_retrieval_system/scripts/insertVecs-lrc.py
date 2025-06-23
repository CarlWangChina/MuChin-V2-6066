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
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="lrc", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="songName", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="songid", dtype=DataType.INT64),
        FieldSchema(name="dbid", dtype=DataType.INT64),
    ]
    schema = CollectionSchema(fields)
    c = Collection(collection_name, schema)
    id = 0
    for it in entities:
        data = [
            [id],  # id
            [it["info"][3].split("/")[-1].replace("_src.mp3","")],  # path
            [it["encode"].view(-1).tolist()],  # embeddings
            [it["lrc"][0:4096]],  # lrc
            [it["info"][1][0:512]],  # songName
            [int(it["info"][2])],  # songid
            [int(it["info"][0])]  # dbid
        ]
        c.insert(data)
        id += 1
    print("insert",collection_name,"num:",id)
    c.flush()
    c.create_index("embeddings", {
        "metric_type":"L2",
        "index_type":"IVF_FLAT",
        "params":{"nlist":1024}
    })

entities = []
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
for oripath in traverse_folders('/export/data/lrc-vec/encode/'):
    data = torch.load(oripath)
    entities.append(data)
print("create collections")
create_collect("cb_collect_lrc_rob", entities)