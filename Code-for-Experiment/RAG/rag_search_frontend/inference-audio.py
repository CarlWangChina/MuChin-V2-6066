import ClapProcessor
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

host = {
    "dist_backend": "gloo",
    "master_addr": "127.0.0.1",
    "master_port": 25681
}
input_dir = "/export/data/datasets-mp3/"
output_dir = "/export/data/clap/encode/audio/"
state_dir = "/export/data/clap/state/audio/"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _setup(rank: int, host: dict, world_size: int):
    os.environ['MASTER_ADDR'] = host["master_addr"]
    os.environ['MASTER_PORT'] = str(host["master_port"])
    dist.init_process_group(backend=host["dist_backend"],
                            rank=rank,
                            world_size=world_size)

def _cleanup():
    dist.destroy_process_group()

def checkDir(file_path):
    parent_directory = os.path.dirname(file_path)
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)

pathes = []

def get_dir_pathes(input_dir: str):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav") or file.endswith(".mp3"):
                path = os.path.join(root, file)
                pathes.append(path)
                relative_path = os.path.relpath(path, input_dir)
                checkDir(os.path.abspath(os.path.join(output_dir, relative_path)))
                checkDir(os.path.abspath(os.path.join(state_dir, relative_path)))

get_dir_pathes("/export/data/datasets-mp3/cb/")

def processFile(enc, batch):
    if os.path.exists(batch["state_path"]):
        return
    res = dict()
    res["full"] = torch.tensor(enc.processAudio(batch["audio_data"]))
    res["audio_len"] = batch["audio_data"].shape
    sample_n = [4, 6, 8, 10, 12]
    for i in sample_n:
        audios, slice_index = audioSampler.sampleAudioBySecond(batch["audio_data"], sample_rate=batch["sampling_rate"], n=i)
        silce_res = []
        for slice in audios:
            silce_res.append(enc.processAudio(slice))
        silce_tensor = torch.tensor(silce_res)
        res[f"{i}"] = {"ori": silce_tensor, "mean": silce_tensor.mean(dim=0), "index": slice_index}
    torch.save(res, batch["out_path"].with_suffix(".pt"))
    with open(batch["state_path"], "w") as f:
        pass

def process(rank: int, world_size: int, host: dict, data_loader_num_workers: int = 2):
    _setup(rank, host, world_size)
    count = torch.cuda.device_count()
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    print("rank", rank, device)
    ds = dataset.AudioDataset(pathes=pathes,
                              rank=rank,
                              rootPath=input_dir,
                              statePath=state_dir,
                              outputPath=output_dir,
                              getLength=True)
    data_sampler = DistributedSampler(ds,
                                      num_replicas=world_size,
                                      rank=rank)
    data_loader = torch.utils.data.DataLoader(ds,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=data_loader_num_workers,
                                              collate_fn=dataset.AudioDataset.collate_fn,
                                              sampler=data_sampler)
    enc = ClapProcessor.ClapProcessor(device=device)
    for batch in tqdm(data_loader, desc=f"Rank {rank}"):
        if "error" not in batch:
            try:
                processFile(enc, batch)
            except Exception as err:
                print(err)
                traceback.print_exc()
    _cleanup()

if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        logger.warning("CUDA_VISIBLE_DEVICES is not set.  Using all available GPUs.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        logger.info(f"Using {world_size} CUDA GPUs.")
        torch.multiprocessing.spawn(process,
                                    args=(world_size, host),
                                    join=True,
                                    nprocs=world_size)