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
    logger.info("Tokenizing acoustic features ...")
    parallel_enabled = torch.cuda.is_available()
    configs = OmegaConf.load(current_path / CONFIG_FILE)
    configs = configs.add_clap
    configs.current_path = str(current_path)
    class_name = "ama-prof-divi.pre_processing.add_clap_vectors.AddClapVectors"
    start_parallel_processing(processor_class_name=class_name,
                              processor_entry_point="process",
                              processor_class_init_kwargs={"configs": OmegaConf.to_container(configs)},
                              parallel_enabled=parallel_enabled,
                              parallel_backend=configs["parallelism"]["dist_backend"],
                              master_addr=configs["parallelism"]["master_addr"],
                              master_port=configs["parallelism"]["master_port"])