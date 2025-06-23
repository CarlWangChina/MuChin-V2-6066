import RobertaProcessor
import dataset
import os
import json
import torch
from Code_for_Experiment.Metrics.music_understanding_model.apex.apex.parallel import distributed as dist
from torch.utils.data.distributed import DistributedSampler
import logging
import audioSampler
import traceback
from pathlib import Path
from tqdm import tqdm

host = {"dist_backend": "gloo", "master_addr": "127.0.0.1", "master_port": 25681}
input_dir = "/export/data/lrc-vec/lyrics/"
output_dir = "/export/data/lrc-vec/encode/"
state_dir = "/export/data/lrc-vec/states/"
audio_dir = "/export/data/datasets-mp3/cb/"
error_dir = "/export/data/lrc-vec/error/"
index_file = "/home/carl/lyric2vector/cb_all_songname_69433.csv"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _setup(rank: int, host: dict, world_size: int):
    os.environ['MASTER_ADDR'] = host["master_addr"]
    os.environ['MASTER_PORT'] = str(host["master_port"])
    dist.init_process_group(backend=host["dist_backend"], rank=rank, world_size=world_size)

def _cleanup():
    dist.destroy_process_group()

def processFile(enc, batch):
    if os.path.exists(batch["state_path"]):
        return
    res = dict()
    res["lrc"] = batch["lrc_data"]
    res["info"] = batch["info"]
    res["encode"] = enc.processText([batch["lrc_data"]])
    torch.save(res, batch["out_path"].with_suffix(".pt"))
    with open(batch["state_path"], "w") as f:
        pass

def process(rank: int, world_size: int, host: dict, data_loader_num_workers: int = 2):
    _setup(rank, host, world_size)
    count = torch.cuda.device_count()
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    print("rank", rank, device)
    ds = dataset.LrcDataset(
        indexPathes=index_file,
        rank=rank,
        rootPath=input_dir,
        statePath=state_dir,
        audioPath=audio_dir,
        outputPath=output_dir
    )
    data_sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank)
    data_loader = torch.utils.data.DataLoader(ds,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=data_loader_num_workers,
                                              collate_fn=dataset.AudioDataset.collate_fn,
                                              sampler=data_sampler)
    enc = RobertaProcessor.RobertaProcessor(device=device)
    for batch in tqdm(data_loader, desc=f"Rank {rank}"):
        if "error" not in batch:
            try:
                processFile(enc, batch)
            except Exception as err:
                torch.save((err, batch), f"{error_dir}/{batch['info'][0]}.pt")
    _cleanup()

if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        logger.warning("CUDA_VISIBLE_DEVICES is not set.  Using all available GPUs.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        logger.info(f"Using {world_size} CUDA GPUs.")
        torch.multiprocessing.spawn(process, args=(world_size, host), join=True, nprocs=world_size)