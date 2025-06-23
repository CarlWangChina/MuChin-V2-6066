import torch
import deepspeed
from pathlib import Path
from typing import Optional, Tuple
from ama_prof_divi_common.data import RandomChunkedMMapDataset, MMapDatasetConfig, MMapDatasetGroupConfig
from ama_prof_divi_common.utils import get_hparams
import ama_prof_divi_common.utils.dist_wrapper as dists
from Code_for_Experiment.Targeted_Training.hifigan_vocoder.ama-prof-divi_hifigan.mel import MelGenerator
NUM_LOCAL_RANK = 8

class ChunkedWavDataset(RandomChunkedMMapDataset):
    def __init__(self, *, meta_file: str, data_file: str, chunk_size: int, start_proportion: Optional[float] = None, end_proportion: Optional[float] = None):
        assert Path(meta_file).exists(), f"Meta file {meta_file} does not exist."
        assert Path(data_file).exists(), f"Data file {data_file} does not exist."
        group_config = MMapDatasetGroupConfig(
            group_name="wav",
            data_file=data_file,
            column_offset=2,
            column_length=4,
            dimensions=torch.Size((1,)),
            sample_rate=chunk_size,
            dtype=torch.int16
        )
        ds_config = MMapDatasetConfig(
            meta_file=meta_file,
            groups=[group_config],
            column_id=1
        )
        super(ChunkedWavDataset, self).__init__(
            ds_config,
            chunk_duration=1.0,
            start_proportion=start_proportion,
            end_proportion=end_proportion
        )
        self.chunk_size = chunk_size
        root_dir = Path(__file__).parent.parent.parent
        hparams = get_hparams(root_dir)
        mel_cfg = hparams.mel_default
        self.mel_generator = MelGenerator(**mel_cfg)

    @staticmethod
    def make_datasets(*, meta_file: Optional[str] = None, data_file: Optional[str] = None, chunk_size: Optional[int] = None, use_data_proportions: Optional[float] = None, train_proportion: Optional[float] = None) -> Tuple["ChunkedWavDataset", "ChunkedWavDataset"]:
        root_dir = Path(__file__).parent.parent.parent
        hparams = get_hparams(root_dir)
        cfg = hparams.training.dataset
        rank = dists.get_rank()
        logger = deepspeed.logger
        logger.info("Making training/validation datasets for rank %d", rank)
        if meta_file is None:
            meta_file = cfg.data_dir + "/" + (cfg.meta_file_pattern % (rank // NUM_LOCAL_RANK))
        if data_file is None:
            data_file = cfg.data_dir + "/" + (cfg.data_file_pattern % (rank // NUM_LOCAL_RANK))
        chunk_size = chunk_size or cfg.chunk_size
        use_data_proportions = use_data_proportions or cfg.use_data_proportions
        train_proportion = train_proportion or cfg.train_proportion
        training_start_prop = 0.0
        training_end_prop = train_proportion * use_data_proportions
        validation_start_prop = training_end_prop
        validation_end_prop = use_data_proportions
        return (
            ChunkedWavDataset(
                meta_file=meta_file,
                data_file=data_file,
                chunk_size=chunk_size,
                start_proportion=training_start_prop,
                end_proportion=training_end_prop
            ),
            ChunkedWavDataset(
                meta_file=meta_file,
                data_file=data_file,
                chunk_size=chunk_size,
                start_proportion=validation_start_prop,
                end_proportion=validation_end_prop
            )
        )

    def __getitem__(self, item):
        data_item = super().__getitem__(item)
        wav_data = data_item["wav"].float() / 32768.0
        data_item["wav"] = wav_data
        data_item["mel"] = self.mel_generator(wav_data.unsqueeze(0)).squeeze(0).squeeze(0)
        return data_item