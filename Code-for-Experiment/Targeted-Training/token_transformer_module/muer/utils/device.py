import torch
from ama_prof_divi.utils import logging

logger = logging.getLogger(__name__)

def probe_devices() -> (str, [int]):
    device = "cpu"
    device_ids = []

    logger.info("Probing devices...")
    if torch.cuda.is_available():
        device = "cuda"
        for i in range(torch.cuda.device_count()):
            device_ids.append(i)
    elif torch.backends.mps.is_available():
        device = "mps"
    logger.info(f"Device probed: {device}")
    return device, device_ids