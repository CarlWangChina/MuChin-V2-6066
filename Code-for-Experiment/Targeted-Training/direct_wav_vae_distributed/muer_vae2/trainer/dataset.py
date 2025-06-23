import torch
import deepspeed
import ray
import os
import shutil
from pathlib import Path
from pyiceberg.catalog import load_catalog
from ray.data.dataset import MaterializedDataset
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.hparams import get_hparams
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.dist_wrapper import dist
from pyiceberg.expressions import (
    And,
    GreaterThanOrEqual,
    LessThanOrEqual,
    NotNaN
)

logger = deepspeed.logger
PARQUET_FILE = "/tmp/vae_ds.parquet"

def build_vae_dataset() -> MaterializedDataset:
    root_dir = Path(__file__).parent.parent.parent
    hparams = get_hparams(root_dir)
    local_rank = dist.get_local_rank()
    if local_rank == 0:
        catalog_properties = {
            "uri": os.environ.get("CATALOG_ENDPOINT"),
            "s3.endpoint": os.environ.get("AWS_S3_ENDPOINT"),
            "s3.access-key-id": os.environ.get("AWS_ACCESS_KEY_ID"),
            "s3.secret-access-key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
            "s3.region": os.environ.get("AWS_REGION"),
        }
        catalog = load_catalog("rest-catalog", **catalog_properties)
        table = catalog.load_table(hparams.training.dataset.table_name)
        logger.info("Scanning the pyarrow table...")
        min_duration = (hparams.vae.chunk_length * 1.2) / hparams.vae.sampling_rate
        pa_table = table.scan(selected_fields=hparams.training.dataset.columns,
                              row_filter=And(
                                  NotNaN("duration"),
                                  GreaterThanOrEqual("duration", min_duration)
                              )).to_arrow()
        logger.info("Scanning the dataset...")
        ds = ray.data.from_arrow(pa_table)
        logger.info("Global shuffling the dataset...")
        ds = ds.random_shuffle(seed=hparams.training.trainer.random_seed)
        logger.info("Global shuffling done.")
        ds = ds.materialize()
        if os.path.exists(PARQUET_FILE):
            shutil.rmtree(PARQUET_FILE)
            assert not os.path.exists(PARQUET_FILE)
        ds.write_parquet(PARQUET_FILE)
    dist.barrier()
    if local_rank != 0:
        ds = ray.data.read_parquet(PARQUET_FILE)
        ds = ds.materialize()
    dist.barrier()
    return ds