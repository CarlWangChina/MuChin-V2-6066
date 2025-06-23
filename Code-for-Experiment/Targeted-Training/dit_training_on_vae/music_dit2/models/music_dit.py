import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict
from ama_prof_divi_common.utils import get_logger, get_hparams
from ama_prof_divi_text.phoneme import PhonemeTokenizer
from Code_for_Experiment.Targeted_Training.dit_training_on_vae.music_dit2.modules.embeddings.label_embedding import LabelEmbedding
from music_dit2.modules.diffusion import DiT, DDIMSampler, TrainingLoss, Diffusion
logger = get_logger(__name__)

class MusicDiTModel(nn.Module):
    def __init__(self):
        super(MusicDiTModel, self).__init__()
        root_dir = Path(__file__).parent.parent.parent
        self.hparams = get_hparams(root_dir).music_dit
        self.dit_model = DiT(
            input_dim=self.hparams.input_dim,
            hidden_dim=self.hparams.hidden_dim,
            num_layers=self.hparams.num_layers,
            num_heads=self.hparams.num_heads,
            dropout=self.hparams.dropout,
            use_causality=self.hparams.use_causality,
            use_cross_attention=self.hparams.use_cross_attention,
            use_rpr=self.hparams.use_rpr,
            context_dim=self.hparams.context_dim,
            pos_embedding=self.hparams.pos_embedding,
            max_position=self.hparams.max_position,
            use_learned_variance=self.hparams.use_learned_variance
        )
        self.ddim_sampler = DDIMSampler(
            beta_start=self.hparams.sampler.beta_start,
            beta_end=self.hparams.sampler.beta_end,
            beta_schedule=self.hparams.sampler.beta_schedule,
            timestep_spacing=self.hparams.sampler.timestep_spacing,
            num_training_timesteps=self.hparams.sampler.num_training_timesteps,
            dynamic_thresholding_ratio=self.hparams.sampler.dynamic_thresholding_ratio,
            clip_sample_range=self.hparams.sampler.clip_sample_range
        )
        self.num_inference_timesteps = self.hparams.sampler.num_inference_timesteps
        self.ddim_sampler.set_inference_timesteps(self.num_inference_timesteps)
        self.training_loss = TrainingLoss(
            sampler=self.ddim_sampler,
            loss_type=self.hparams.loss.loss_type,
            mse_loss_weight=self.hparams.loss.mse_loss_weight,
            vb_loss_weight=self.hparams.loss.vb_loss_weight
        )
        self.diffusion = Diffusion(
            model=self.dit_model,
            sampler=self.ddim_sampler,
            training_loss=self.training_loss
        )
        self.clap_embedding = nn.Linear(self.hparams.clap_dim, self.hparams.hidden_dim)
        phoneme_tokenizer = PhonemeTokenizer()
        self.lyrics_embedding = LabelEmbedding(num_classes=phoneme_tokenizer.vocab_size, hidden_dim=self.hparams.context_dim)
        self.input_dim = self.hparams.input_dim
        self.hidden_dim = self.hparams.hidden_dim
        self.clap_dim = self.hparams.clap_dim
        self.lyrics_vocab_size = phoneme_tokenizer.vocab_size
        self.vae_to_clap_ratio = self.hparams.vae_to_clap_ratio
        self.init_parameters_()

    def init_parameters_(self):
        self.diffusion.initialize_weights()
        nn.init.xavier_uniform_(self.clap_embedding.weight)
        nn.init.zeros_(self.clap_embedding.bias)

    def _align_clap_to_vae(self, clap: torch.Tensor, seq_len: int) -> torch.Tensor:
        batch_size = clap.size(0)
        num_frames = clap.size(1)
        clap_out = torch.zeros(batch_size, seq_len, clap.size(-1), device=clap.device).type_as(clap)
        for i in range(seq_len):
            idx = min(i // self.vae_to_clap_ratio, num_frames - 1)
            clap_out[:, i, :] = clap[:, idx, :]
        return clap_out

    def _preprocess_input(self, *, vae: torch.Tensor, vae_mask: Optional[torch.Tensor] = None, clap: Optional[torch.Tensor] = None, clap_mask: Optional[torch.Tensor] = None, lyrics: Optional[torch.Tensor] = None, lyrics_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        assert vae.dim() == 3, f"VAE must be 3D tensor, but got {vae.dim()}"
        assert vae.size(-1) == self.input_dim, f"Input dimension mismatch: {self.input_dim} vs {vae.size(-1)}"
        batch_size = vae.size(0)
        seq_len = vae.size(1)
        if vae_mask is not None:
            assert vae_mask.dim() == 2, f"VAE mask must be 2D tensor, but got {vae_mask.dim()}"
            assert vae_mask.size(0) == batch_size, f"Batch size mismatch: {batch_size} vs {vae_mask.size(0)}"
            assert vae_mask.size(1) == vae.size(1), f"Sequence length mismatch: {vae.size(1)} vs {vae_mask.size(1)}"
        if clap is not None:
            assert clap.dim() == 3, f"Clap must be 3D tensor, but got {clap.dim()}"
            assert clap.size(0) == batch_size, f"Batch size mismatch: {batch_size} vs {clap.size(0)}"
            assert clap.size(-1) == self.clap_dim, f"Clap dimension mismatch: {self.clap_dim} vs {clap.size(-1)}"
            if clap_mask is not None:
                assert clap_mask.dim() == 2, f"Clap mask must be 2D tensor, but got {clap_mask.dim()}"
                assert clap_mask.size(0) == batch_size, f"Batch size mismatch: {batch_size} vs {clap_mask.size(0)}"
                assert clap_mask.size(1) == clap.size(1), f"Clap length mismatch: {clap.size(1)} vs {clap_mask.size(1)}"
        if lyrics is not None and lyrics.numel() == 0:
            lyrics = None
            lyrics_mask = None
        if lyrics is not None:
            assert lyrics.dim() == 2, f"Lyrics must be 2D tensor, but got {lyrics.dim()}"
            assert lyrics.size(0) == batch_size, f"Batch size mismatch: {batch_size} vs {lyrics.size(0)}"
            assert 0 <= lyrics.min() <= lyrics.max() < self.lyrics_vocab_size, f"Lyrics out of range: {lyrics.min()}, {lyrics.max()}"
            if lyrics_mask is not None:
                assert lyrics_mask.dim() == 2, f"Lyrics mask must be 2D tensor, but got {lyrics_mask.dim()}"
                assert lyrics_mask.size(0) == batch_size, f"Batch size mismatch: {batch_size} vs {lyrics_mask.size(0)}"
                assert lyrics_mask.size(1) == lyrics.size(1), f"Lyrics length mismatch: {lyrics.size(1)} vs {lyrics_mask.size(1)}"
        if clap is not None:
            if clap_mask is not None:
                clap = clap.masked_fill(clap_mask.unsqueeze(-1) == False, 0.0)
            clap_emb = self._align_clap_to_vae(self.clap_embedding(clap), seq_len)
        else:
            clap_emb = None
        if lyrics is not None:
            lyrics_emb = self.lyrics_embedding(lyrics)
        else:
            lyrics_emb = None
        return {
            "samples": vae,
            "conditions": clap_emb,
            "padding_mask": vae_mask,
            "context": lyrics_emb,
            "context_mask": lyrics_mask
        }

    def training_step(self, *, vae: torch.Tensor, vae_mask: Optional[torch.Tensor] = None, clap: Optional[torch.Tensor] = None, clap_mask: Optional[torch.Tensor] = None, lyrics: Optional[torch.Tensor] = None, lyrics_mask: Optional[torch.Tensor] = None, rank: int = 0, world_size: int = 1) -> Dict[str, torch.Tensor]:
        inp = self._preprocess_input(vae=vae, vae_mask=vae_mask, clap=clap, clap_mask=clap_mask, lyrics=lyrics, lyrics_mask=lyrics_mask)
        loss_dict = self.diffusion.training_step(
            samples=inp["samples"],
            conditions=inp["conditions"],
            padding_mask=inp["padding_mask"],
            context=inp["context"],
            context_mask=inp["context_mask"],
            rank=rank,
            world_size=world_size
        )
        return loss_dict

    def forward(self, vae: torch.Tensor, vae_mask: Optional[torch.Tensor] = None, clap: Optional[torch.Tensor] = None, clap_mask: Optional[torch.Tensor] = None, lyrics: Optional[torch.Tensor] = None, lyrics_mask: Optional[torch.Tensor] = None, rank: int = 0, world_size: int = 1) -> Dict[str, torch.Tensor]:
        return self.training_step(vae=vae, vae_mask=vae_mask, clap=clap, clap_mask=clap_mask, lyrics=lyrics, lyrics_mask=lyrics_mask, rank=rank, world_size=world_size)

    @torch.inference_mode()
    def inference(self, *, x: torch.Tensor, mask: Optional[torch.Tensor] = None, clap: Optional[torch.Tensor] = None, clap_mask: Optional[torch.Tensor] = None, lyrics: Optional[torch.Tensor] = None, lyrics_mask: Optional[torch.Tensor] = None, start_timestep_index: int = 0, cfg_scale: Optional[float] = None, eta: float = 0.0) -> torch.Tensor:
        inp = self._preprocess_input(vae=x, vae_mask=mask, clap=clap, clap_mask=clap_mask, lyrics=lyrics, lyrics_mask=lyrics_mask)
        generated = self.diffusion.generate(
            x=inp["samples"],
            conditions=inp["conditions"],
            padding_mask=inp["padding_mask"],
            context=inp["context"],
            context_mask=inp["context_mask"],
            start_timestep_index=start_timestep_index,
            cfg_scale=cfg_scale,
            eta=eta
        )
        return generated

    @torch.inference_mode()
    def test_inference(self, *, vae: torch.Tensor, vae_mask: Optional[torch.Tensor] = None, clap: Optional[torch.Tensor] = None, clap_mask: Optional[torch.Tensor] = None, lyrics: Optional[torch.Tensor] = None, lyrics_mask: Optional[torch.Tensor] = None, rank: int = 0, world_size: int = 1) -> torch.Tensor:
        inp = self._preprocess_input(vae=vae, vae_mask=vae_mask, clap=clap, clap_mask=clap_mask, lyrics=lyrics, lyrics_mask=lyrics_mask)
        inference_loss = self.diffusion.test_inference(samples=inp["samples"], conditions=inp["conditions"], padding_mask=inp["padding_mask"], context=inp["context"], context_mask=inp["context_mask"], cfg_scale=0.5, rank=rank, world_size=world_size)
        return inference_loss

    @torch.inference_mode()
    def shallow_diffusion(self, vae_ref: torch.Tensor, mask: Optional[torch.Tensor] = None, clap: Optional[torch.Tensor] = None, clap_mask: Optional[torch.Tensor] = None, lyrics: Optional[torch.Tensor] = None, lyrics_mask: Optional[torch.Tensor] = None, start_timestep_index: int = 0, cfg_scale: Optional[float] = None, eta: float = 0.0) -> torch.Tensor:
        inp = self._preprocess_input(vae=vae_ref, vae_mask=mask, clap=clap, clap_mask=clap_mask, lyrics=lyrics, lyrics_mask=lyrics_mask)
        generated = self.diffusion.shallow_diffusion(
            ref_samples=inp["samples"],
            conditions=inp["conditions"],
            padding_mask=inp["padding_mask"],
            context=inp["context"],
            context_mask=inp["context_mask"],
            start_timestep_index=start_timestep_index,
            cfg_scale=cfg_scale,
            eta=eta
        )
        return generated