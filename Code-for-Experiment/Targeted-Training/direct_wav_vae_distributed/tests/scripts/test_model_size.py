import unittest
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
from ama_prof_divi_vae2.model import VAE2Model

logger = get_logger(__name__)

class TestModuleSize(unittest.TestCase):
    def test_model_size(self):
        model = VAE2Model()
        encoder_size = sum(p.numel() for p in model.encoder.parameters())
        decoder_size = sum(p.numel() for p in model.decoder.parameters())
        logger.info("test_model_size: encoder_size: %s, decoder_size: %s", encoder_size, decoder_size)