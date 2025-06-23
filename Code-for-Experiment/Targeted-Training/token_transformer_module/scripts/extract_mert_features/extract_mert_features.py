import sys
import torch
from pathlib import Path
from omegaconf import OmegaConf

current_path = Path(__file__).absolute().parent
root_path = current_path.parent.parent
sys.path.append(str(root_path))
sys.path.append(str(current_path))

import get_logger
from ama_prof_divi.utils.parallel import start_parallel_processing

logger = get_logger(__name__)
CONFIG_FILE = "config.yaml"

if __name__ == "__main__":
    logger.info("Extracting mert features ...")
    parallel_enabled = torch.cuda.is_available()
    _configs = OmegaConf.load(current_path / CONFIG_FILE)
    _configs = _configs.mert_extractor
    _configs.root_path = str(root_path)
    _configs.current_path = str(current_path)
    start_parallel_processing(
        processor_class_name="ama-prof-divi.pre_processing.mert_extractor.MertExtractor",
        processor_entry_point="process",
        processor_class_init_kwargs={"configs": OmegaConf.to_container(_configs)},
        parallel_enabled=parallel_enabled,
        parallel_backend=_configs["dist_backend"],
        master_addr=_configs["master_addr"],
        master_port=_configs["master_port"],
    )