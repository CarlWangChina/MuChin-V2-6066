import torch
import deepspeed
from typing import Dict, Optional
from pathlib import Path
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.hparams import get_hparams
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils import dist as dists
from ama_prof_divi_common.data import MMapDataset, MMapDatasetConfig, MMapDatasetGroupConfig

NUM_LOCAL_RANKS = 8

class DiTDataset(MMapDataset):
    """Dataset for training DiT model.
    Args:
        meta_file (str): Path to the meta file.
        data_files (Dict[str, str]): Path to the data files.
        column_ids (Dict[str, int]): Column ID dictionary in the meta file.
        dimensions (Dict[str, str]): Dimension dictionary of each group.
        data_types (Dict[str, str]): Data type dictionary of each group.
        rank (int): Rank of the current process.
        start_proportion (Optional[float]): Start proportion of the dataset.
        end_proportion (Optional[float]): End proportion of the dataset.
    """
    def __init__(self,
                 meta_file: str,
                 data_files: Dict[str, str],
                 column_ids: Dict[str, int],
                 dimensions: Dict[str, str],
                 data_types: Dict[str, str],
                 rank: int,
                 start_proportion: Optional[float] = None,
                 end_proportion: Optional[float] = None):
        assert Path(meta_file).exists(), f"Meta file {meta_file} does not exist."
        assert len(data_files) > 0, "Data files are empty."
        for data_file in data_files.values():
            data_file_t = data_file % (rank // NUM_LOCAL_RANKS)
            assert Path(data_file_t).exists(), f"Data file {data_file_t} does not exist."
        group_configs = [
            MMapDatasetGroupConfig(
                group_name=group_name,
                data_file=data_file % (rank // NUM_LOCAL_RANKS),
                column_offset=column_ids[group_name + "_offset"],
                column_length=column_ids[group_name + "_length"],
                dimensions=eval(dimensions[group_name]),
                dtype=eval("torch." + data_types[group_name])
            )
            for group_name, data_file in data_files.items()
        ]
        ds_config = MMapDatasetConfig(
            meta_file=meta_file,
            groups=group_configs,
            column_id=column_ids["song_id"],
            node_id=rank // NUM_LOCAL_RANKS,
            column_node_id=column_ids["node"],
            skip_csv_header=True)
        super(DiTDataset, self).__init__(ds_config,
                                         start_proportion=start_proportion,
                                         end_proportion=end_proportion)

    @staticmethod
    def make_datasets():
        """Make training and validation datasets with all configurations read from the hyperparameters.
        """
        root_dir = Path(__file__).parent.parent.parent
        hparams = get_hparams(root_dir).training.dataset
        rank = dists.get_rank()
        logger = deepspeed.logger
        logger.info("Making training and validation datasets for rank %d...", rank)
        training_end_proportion = hparams.use_data_proportion * hparams.train_proportion
        validation_end_proportion = hparams.use_data_proportion
        train_dataset = DiTDataset(
            meta_file=hparams.meta_file,
            data_files=hparams.data_files,
            column_ids=hparams.column_ids,
            dimensions=hparams.dimensions,
            data_types=hparams.data_types,
            rank=rank,
            start_proportion=0.0,
            end_proportion=training_end_proportion)
        validation_dataset = DiTDataset(
            meta_file=hparams.meta_file,
            data_files=hparams.data_files,
            column_ids=hparams.column_ids,
            dimensions=hparams.dimensions,
            data_types=hparams.data_types,
            rank=rank,
            start_proportion=training_end_proportion,
            end_proportion=validation_end_proportion)
        return train_dataset, validation_dataset