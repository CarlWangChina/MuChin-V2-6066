import os
import torch
from torch import nn
from pathlib import Path
from typing import Optional
from ama_prof_divi.utils import download_file
from ama_prof_divi.utils.logging import get_logger
from ama_prof_divi.models.lyrics import get_lyrics_tokenizer
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.modules.rvq.rvq import ResidualVectorQuantization

logger = get_logger(__name__)

class RVQTokenizer(nn.Module):
    def __init__(self, hparams: dict, load_pretrained: bool = True, device: str or torch.device = "cpu"):
        super(RVQTokenizer, self).__init__()
        self.hparams = hparams
        self.rvq_hparams = hparams["ama-prof-divi"]["models"]["semantic"]["tokenizer"]["rvq"]
        self.is_fitted = False
        self.dim = self.rvq_hparams["n_dim"]
        self.num_quantizers = self.rvq_hparams["num_quantizers"]
        self.codebook_size = self.rvq_hparams["codebook_size"]
        learnable = self.rvq_hparams["learnable_codebook"]
        self.rvq = ResidualVectorQuantization(dim=self.dim, num_quantizers=self.num_quantizers, codebook_size=self.codebook_size, similarity=self.rvq_hparams["similarity"], kmeans_init=False, channel_last=True, ema_update=not learnable, learnable_codebook=learnable).to(device)
        lyrics_tokenizer = get_lyrics_tokenizer()
        self.start_token = lyrics_tokenizer.start_token
        self.pad_token = lyrics_tokenizer.pad_token
        self.end_token = lyrics_tokenizer.end_token
        self.mask_token = lyrics_tokenizer.mask_token
        self.sep_token = lyrics_tokenizer.sep_token
        self.unknown_token = lyrics_tokenizer.unknown_token
        self.special_tokens_dict = {}
        self.vocab_size = self.codebook_size
        for token in sorted(lyrics_tokenizer.special_tokens_set()):
            self.special_tokens_dict[token] = self.vocab_size
            self.vocab_size += 1
        if load_pretrained:
            self.load_pretrained()
        self.eval()

    def load_pretrained(self):
        root_path = Path(self.hparams["ama-prof-divi"]["root_path"])
        checkpoints_dir = root_path.joinpath("checkpoints").joinpath("mert_rvq")
        if not os.path.exists(checkpoints_dir):
            logger.info(f"Creating directory '{checkpoints_dir}' ...")
            os.makedirs(checkpoints_dir)
        checkpoints_file = checkpoints_dir.joinpath(self.rvq_hparams["pretrained_model"])
        logger.info(f"Downloading pretrained checkpoints '{self.rvq_hparams['pretrained_model']}' ...")
        checksum = None
        if "pretrained_model_sha256" in self.rvq_hparams:
            checksum = self.rvq_hparams["pretrained_model_sha256"]
        download_file(self.rvq_hparams["pretrained_model_url"], str(checkpoints_file), expected_sha256=checksum)
        logger.info("Loading pretrained RVQ tokenizer.")
        state_dict = torch.load(checkpoints_file, map_location="cpu")["state_dict"]
        self.rvq.load_state_dict(state_dict)
        self.is_fitted = True
        logger.info("Vocab size: %d", self.vocab_size)

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.inference_mode()
    def tokenize(self, x: torch.Tensor, *, num_q: Optional[int] = None) -> torch.Tensor:
        assert x.dim() == 2 or x.dim() == 3, ("Input data should be in the shape of (n_batches, n_samples, n_dim), or (n_samples, n_dim).")
        if x.dim() == 2:
            x = x.unsqueeze(0)
        assert x.shape[-1] == self.dim, f"The dimension of input data must have {self.dim}."
        indices = self.rvq(x, num_quantizers=num_q)["all_indices"]
        return indices

    def forward(self, x: torch.Tensor, *, num_q: Optional[int] = None) -> torch.Tensor:
        return self.tokenize(x, num_q=num_q)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        return self.rvq.get_output_from_indices(indices)

    def is_special_token(self, token: int) -> bool:
        return self.codebook_size <= token < self.vocab_size

    def encode_special_token(self, token: str) -> int:
        if token not in self.special_tokens_dict:
            raise ValueError(f"{token} is not a special token.")
        return self.special_tokens_dict[token]

    @property
    def pad_id(self) -> int:
        return self.encode_special_token(self.pad_token)

    @property
    def sep_id(self) -> int:
        return self.encode_special_token(self.sep_token)

    @property
    def start_id(self) -> int:
        return self.encode_special_token(self.start_token)

    @property
    def end_id(self) -> int:
        return self.encode_special_token(self.end_token)

    @property
    def mask_id(self) -> int:
        return self.encode_special_token(self.mask_token)

    @property
    def unknown_id(self) -> int:
        return self.encode_special_token(self.unknown_token)