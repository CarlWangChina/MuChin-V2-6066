import torch
import unittest
from tqdm import tqdm
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.logging import get_logger
import logging
logger = get_logger(__name__)
from ama_prof_divi.models.prompting import get_prompt_encoder
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.models.acoustic.generators.builder import get_accompaniment_generator
from ama_prof_divi.modules.transformers import InferAccelerationCache
logger = get_logger(__name__)

class TestAccompanimentGenerator(unittest.TestCase):
    def setUp(self):
        init_hparams()
        post_init_hparams()
        self.device = get_hparams()["ama_prof_divi"]["device"]
        logger.info("Getting the accompaniment generator model.")
        self.accompaniment_generator = get_accompaniment_generator()
        self.prompt_encoder = get_prompt_encoder()
        self.rvq_vocab_size = self.accompaniment_generator.semantic_rvq_vocab_size
        self.melody_vocab_size = self.accompaniment_generator.melody_vocab_size
        logger.info("Setup OK.")

    def test_build_accompaniment_generator(self):
        logger.info("test_build_accompaniment_generator")
        self.assertIsNotNone(self.accompaniment_generator)

    def _get_test_data(self, num_batches: int = 1, rvq_seq_len: int = 20, acoustic_seq_len: int = 320) -> dict:
        text_prompts = ["美妙的中文歌曲"]
        prompts, _ = self.prompt_encoder.get_text_prompt_embeddings(text_prompt=text_prompts, text_prompt_language='zh')
        prompts = prompts.repeat(num_batches, 1).to(self.device)
        self.assertEqual(prompts.shape, (num_batches, self.prompt_encoder.joint_embedding_dim))
        rvq_tokens = torch.randint(0, self.rvq_vocab_size, (num_batches, rvq_seq_len, self.accompaniment_generator.semantic_rvq_num_q)).to(self.device)
        melody_tokens = torch.randint(0, self.melody_vocab_size, (num_batches, rvq_seq_len)).to(self.device)
        ss_tokens = torch.randint(0, self.accompaniment_generator.ss_vocab_size, (num_batches, rvq_seq_len)).to(self.device)
        acoustic_tokens = torch.randint(0, self.accompaniment_generator.vocab_size, (num_batches, acoustic_seq_len)).to(self.device)
        return {
            "prompts": prompts,
            "rvq_tokens": rvq_tokens,
            "melody_tokens": melody_tokens,
            "ss_tokens": ss_tokens,
            "acoustic_tokens": acoustic_tokens
        }

    def test_generate(self):
        logger.info("test_generate")
        self.accompaniment_generator.eval()
        cache = InferAccelerationCache(self.accompaniment_generator.transformer_args)
        test_data = self._get_test_data()
        acoustic_tokens = self.accompaniment_generator.generate(rvq_tokens=test_data["rvq_tokens"], melody_tokens=test_data["melody_tokens"], ss_tokens=test_data["ss_tokens"], prompt=test_data["prompts"], cache=cache)
        self.assertIsNotNone(acoustic_tokens)
        logger.info("acoustic_tokens: %s", acoustic_tokens)

    def test_training_transformer(self):
        logger.info("test_training_transformer")
        self.accompaniment_generator.train()
        test_data = self._get_test_data(rvq_seq_len=10, acoustic_seq_len=100)
        num_windows = self.accompaniment_generator.get_num_windows(test_data["acoustic_tokens"].shape[1])
        self.assertGreater(num_windows, 0)
        logger.info("test_training_transformer: num_windows: %s", num_windows)
        for i in tqdm(range(num_windows)):
            self.accompaniment_generator.forward_training_transformer(rvq_tokens=test_data["rvq_tokens"], melody_tokens=test_data["melody_tokens"], ss_tokens=test_data["ss_tokens"], acoustic_tokens=test_data["acoustic_tokens"], start_window=i, num_windows=num_windows)

    def test_training_diffusion(self):
        logger.info("test_training_diffusion")
        self.accompaniment_generator.train_diffusion()
        test_data = self._get_test_data()
        loss = self.accompaniment_generator.forward_training_diffusion(acoustic_tokens=test_data["acoustic_tokens"], prompt_embedding=test_data["prompts"])
        logger.info("loss: %s", loss)

    def test_training_controlnet(self):
        logger.info("test_training_controlnet")
        if not self.accompaniment_generator.controlnet_enabled:
            logger.warning("test_training_controlnet: controlnet is not enabled.")
            return
        self.accompaniment_generator.train_controlnet()
        test_data = self._get_test_data()
        loss = self.accompaniment_generator.forward_training_controlnet(acoustic_tokens=test_data["acoustic_tokens"], prompt_embedding=test_data["prompts"])
        logger.info("loss: %s", loss)

    def test_training_output(self):
        logger.info("test_training_output")
        self.accompaniment_generator.train_output_projector()
        test_data = self._get_test_data()
        loss = self.accompaniment_generator.forward_training_output(acoustic_tokens=test_data["acoustic_tokens"])
        logger.info("loss: %s", loss)

from ama_prof_divi.hparams import init_hparams, post_init_hparams, get_hparams