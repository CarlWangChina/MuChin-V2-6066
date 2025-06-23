import unittest
import torch
import torch.nn.functional as F
from ama_prof_divi.utils import logging
from ama_prof_divi.configs import init_hparams, post_init_hparams
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.modules.rvq.rvq import ResidualVectorQuantization
logger = logging.get_logger(__name__)

class TestRVQ(unittest.TestCase):
    def setUp(self):
        init_hparams()
        post_init_hparams()
        logger.info("Setup: building RVQ.")
        self.num_quantizers = 8
        self.dim = 1024
        self.codebook_size = 64
        self.similarity = "cosine"
        self.kmeans_init = True

    def test_rvq_inference(self):
        rvq = ResidualVectorQuantization(dim=self.dim,
                                         num_quantizers=self.num_quantizers,
                                         codebook_size=self.codebook_size,
                                         similarity=self.similarity,
                                         kmeans_init=self.kmeans_init,
                                         channel_last=True)
        rvq.eval()
        x = torch.randn(2, 256, self.dim)
        y = rvq(x, return_all_codes=True)
        logger.info("quantized_out: %s", y["quantized_out"].shape)
        logger.info("all_indices: %s", y["all_indices"].shape)
        logger.info("all_codes: %s", y["all_codes"].shape)
        logger.info("all_losses: %s", y["all_losses"].shape)
        logger.info(rvq.get_output_from_indices(y["all_indices"]).shape)
        recovered_output = rvq.get_output_from_indices(y["all_indices"])
        diff = (recovered_output - y["quantized_out"]).abs().max()
        self.assertLess(diff, 1e-5)
        self.assertLess(y["all_indices"].max(), self.codebook_size)
        loss = F.mse_loss(x, y["quantized_out"])
        logger.info("y.all_losses.mean(): %s", y["all_losses"].mean())
        logger.info("MSE loss: %s", loss)

    def test_rvq_training_ema(self):
        rvq = ResidualVectorQuantization(dim=self.dim,
                                         num_quantizers=self.num_quantizers,
                                         codebook_size=self.codebook_size,
                                         similarity=self.similarity,
                                         kmeans_init=self.kmeans_init,
                                         channel_last=True,
                                         ema_update=True)
        rvq.train()
        x = torch.randn(1, 256, self.dim)
        for i in range(1000):
            y = rvq(x)

    def test_rvq_training_ml(self):
        rvq = ResidualVectorQuantization(dim=self.dim,
                                         num_quantizers=self.num_quantizers,
                                         codebook_size=self.codebook_size,
                                         similarity=self.similarity,
                                         kmeans_init=self.kmeans_init,
                                         channel_last=True,
                                         ema_update=False,
                                         learnable_codebook=True,
                                         in_place_codebook_optimizer=torch.optim.Adam)
        rvq.train()
        x = torch.randn(1, 256, self.dim)
        for i in range(1000):
            y = rvq(x)