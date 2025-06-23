from pymilvus import connections, db, Collection, CollectionSchema, FieldSchema, DataType, utility
import ClapProcessor
import RobertaProcessor
import threading
import torch
import transformers

connections.connect(
    alias="default",
    uri="http://localhost:19530",
    token="root:Milvus",
)
db.using_database("music6w2")
print("existed collections:", utility.list_collections())
processor_clap = ClapProcessor.ClapProcessor()
processor_roberta = RobertaProcessor.RobertaProcessor()
translation_pipeline = transformers.pipeline('translation', model='Helsinki-NLP/opus-mt-zh-en')
lock = threading.Lock()

def search_data_by_audio(text_data: list[str], table: str = "collect_full", translate: bool = True, mixSearch: bool = False):
    assert len(text_data) >= 1
    with lock:
        c = Collection(table)
        c.load()
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        if translate:
            text_data = [translation_pipeline(t)[0]["translation_text"] for t in text_data]
        if len(text_data) == 1:
            text_vec = [processor_clap.processText([text_data[0], text_data[0]])[0].tolist()]
        else:
            text_vec = processor_clap.processText(text_data).tolist()
        if mixSearch:
            text_vec = torch.tensor(text_vec).mean(dim=0, keepdim=True).tolist()
            text_data = [";".join(text_data)]
        result = c.search(text_vec, "embeddings", search_params, limit=30, output_fields=["path"])
        return result, text_data

def search_data_by_lrc(text_data: list[str], table: str = "collect_lrc", translate: bool = True, mixSearch: bool = False):
    assert len(text_data) >= 1
    with lock:
        c = Collection(table)
        c.load()
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        if translate:
            text_data = [translation_pipeline(t)[0]["translation_text"] for t in text_data]
        if len(text_data) == 1:
            text_vec = [processor_roberta.processText([text_data[0], text_data[0]])[0].tolist()]
        else:
            text_vec = processor_roberta.processText(text_data).tolist()
        if mixSearch:
            text_vec = torch.tensor(text_vec).mean(dim=0, keepdim=True).tolist()
            text_data = [";".join(text_data)]
        result = c.search(text_vec, "embeddings", search_params, limit=30, output_fields=["path", "lrc", "songName"])
        return result, text_data

if __name__ == "__main__":
    result, prompt = search_data_by_audio(["test", "你好"])
    print(result)
    print(len(result))