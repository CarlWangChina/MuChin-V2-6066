from typing import Optional

from ama_prof_divi.utils.logging import get_logger
from ama_prof_divi.configs.hparams import get_hparams
from ama_prof_divi.utils.parallel import start_parallel_processing
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.training.trainer_base.trainer_args import TrainerArgs

logger = get_logger(__name__)

def start_trainer(trainer_class_name: str, trainer_args: TrainerArgs, *, device: Optional[str] = None):
    logger.info("Starting the trainer.")
    trainer_init_kwargs = {
        "args": trainer_args,
        "device": device
    }

    if device == "cpu" or device == "mps":
        trainer_args.parallel_enabled = False
    start_parallel_processing(processor_class_name=trainer_class_name,
                              processor_entry_point="train",
                              processor_class_init_kwargs=trainer_init_kwargs,
                              processor_args=None,
                              parallel_enabled=trainer_args.parallel_enabled,
                              parallel_backend=trainer_args.parallel_backend,
                              master_addr=trainer_args.master_addr,
                              master_port=trainer_args.master_port,
                              random_seed=trainer_args.random_seed)