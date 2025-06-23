import os
import random
import numpy as np
import torch
import torch.distributed as dist
from dataclasses import dataclass, asdict
from importlib import import_module
from ama_prof_divi.configs.hparams import init_hparams, post_init_hparams, get_hparams
logger = get_logger(__name__)
MASTER_PORT = 29500

def setup_parallelism(rank: int, world_size: int, backend: str = "nccl", is_master: bool = True):
    if is_master:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    else:
        dist.init_process_group(init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}", backend=backend, rank=rank, world_size=world_size)

def setup_random_seed(random_seed: int):
    torch.manual_seed(random_seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def cleanup_parallelism():
    dist.barrier()
    dist.destroy_process_group()

@dataclass
class ParallelArgs:
    world_size: int
    backend: str
    master_addr: str
    strategy: str = "ddp"
    master_port: int = MASTER_PORT
    rank_start: int = 0
    random_seed: int = 0
    parallel_enabled: bool = True

def _main(local_rank: int, processor_class_name: str, processor_class_init_kwargs: dict, processor_entry_point: str, processor_args: dict, parallel_args: dict):
    init_hparams()
    post_init_hparams()
    hparams = get_hparams()
    parallel_args = ParallelArgs(**parallel_args)
    setup_random_seed(parallel_args.random_seed)
    if processor_class_init_kwargs is not None and "device" in processor_class_init_kwargs:
        if processor_class_init_kwargs["device"] is not None:
            device = torch.device(processor_class_init_kwargs["device"])
            hparams["ama-prof-divi"]["device"] = device
            del processor_class_init_kwargs["device"]
    parallel_enabled = parallel_args.parallel_enabled
    if parallel_enabled:
        assert parallel_args is not None, "parallel_args are not provided when parallel is enabled"
        world_size = parallel_args.world_size
        rank = parallel_args.rank_start + local_rank
        is_master = (parallel_args.rank_start == 0)
        if local_rank == 0:
            if is_master:
                logger.info(f"Starting parallel processing as master. World size: {world_size}, backend: {parallel_args.backend}.")
            else:
                logger.info(f"Starting parallel processing as slave. World size: {world_size}, backend: {parallel_args.backend} Master is tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
        setup_parallelism(rank=rank, world_size=parallel_args.world_size, backend=parallel_args.backend, is_master=is_master)
        if hparams["ama-prof-divi"]["device"].type == "cuda":
            hparams["ama-prof-divi"]["device"] = torch.device(f"cuda:{local_rank}")
    else:
        world_size = 1
        rank = 0
        is_master = True
    try:
        if "." not in processor_class_name:
            raise ValueError(f"Invalid trainer class name: {processor_class_name}.  Should be module_name.class_name.")
        module_path, class_name = processor_class_name.rsplit('.', 1)
        module = import_module(module_path)
        processor_class = getattr(module, class_name)
        if processor_class_init_kwargs is not None:
            processor = processor_class(**processor_class_init_kwargs, device=hparams["ama-prof-divi"]["device"], parallel_enabled=parallel_enabled, world_size=world_size, rank=rank, local_rank=local_rank, is_master=is_master)
        else:
            processor = processor_class(device=hparams["ama-prof-divi"]["device"], parallel_enabled=parallel_enabled, world_size=world_size, rank=rank, local_rank=local_rank, is_master=is_master)
        if parallel_enabled:
            dist.barrier()
        entry_point = getattr(processor, processor_entry_point)
        if processor_args is not None:
            entry_point(**processor_args)
        else:
            entry_point()
    finally:
        if parallel_enabled:
            dist.barrier()
            cleanup_parallelism()
        if rank == 0:
            logger.info("Processor is done -- Succeeded.")

def start_parallel_processing(processor_class_name: str, processor_entry_point: str, processor_class_init_kwargs: dict = None, processor_args: dict = None, parallel_enabled: bool = True, parallel_backend: str = "nccl", parallel_strategy: str = "ddp", master_addr: str = "localhost", master_port: int = MASTER_PORT, random_seed: int = 0):
    if "MASTER_ADDR" in os.environ:
        if os.environ['MASTER_ADDR'] != master_addr:
            logger.warning(f"MASTER_ADDR is set in the environment variables.  The environment variable '{master_addr}' will prevail.")
        master_addr = os.environ['MASTER_ADDR']
    else:
        os.environ['MASTER_ADDR'] = master_addr
    if "MASTER_PORT" in os.environ:
        if os.environ['MASTER_PORT'] != str(master_port):
            logger.warning(f"MASTER_PORT is set in the environment variables.  The environment variable {master_port} will prevail.")
        master_port = int(os.environ['MASTER_PORT'])
    else:
        os.environ['MASTER_PORT'] = str(master_port)
    if parallel_enabled:
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            logger.warning("CUDA_VISIBLE_DEVICES is not set.  Using all available GPUs.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
        assert torch.cuda.is_available(), "CUDA is not available.  Could not enable parallel processing."
        if "WORLD_SIZE" not in os.environ:
            world_size = torch.cuda.device_count()
            os.environ["WORLD_SIZE"] = str(world_size)
            rank_start = 0
            logger.warning(f"WORLD_SIZE is not set.  Assuming using single node with {world_size} GPUs. For multi-node parallelism, set WORLD_SIZE and RANK_START in the environment variables.")
        else:
            world_size = int(os.environ["WORLD_SIZE"])
            assert "RANK_START" in os.environ, ("RANK_START is not set.  When WORLD_SIZE is used for multi-node training, RANK_START must be also set for the starting point of rank on the current node.")
            rank_start = int(os.environ["RANK_START"])
            logger.info(f"WORLD_SIZE is set.  Assuming multi-node training. World size is {world_size}, local rank start from {rank_start}.")
        logger.info("Starting parallel processing...")
        parallel_args = ParallelArgs(world_size=world_size, rank_start=rank_start, backend=parallel_backend, master_addr=master_addr, master_port=master_port, random_seed=random_seed)
        torch.multiprocessing.spawn(_main, args=(processor_class_name, processor_class_init_kwargs, processor_entry_point, processor_args, asdict(parallel_args)), join=True, nprocs=torch.cuda.device_count())
    else:
        logger.info("Starting non-parallel processing...")
        parallel_args = ParallelArgs(world_size=1, backend=parallel_backend, master_addr=master_addr, strategy=parallel_strategy, master_port=master_port, random_seed=random_seed, parallel_enabled=parallel_enabled)
        _main(0, processor_class_name, processor_class_init_kwargs, processor_entry_point, processor_args, asdict(parallel_args))