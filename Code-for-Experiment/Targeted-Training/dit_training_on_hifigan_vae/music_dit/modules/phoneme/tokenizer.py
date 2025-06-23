import torch
import torch.nn as nn
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger

logger = get_logger(__name__)

class PhonemeTokenizer(nn.Module):
    def __init__(self):
        super(PhonemeTokenizer, self).__init__()