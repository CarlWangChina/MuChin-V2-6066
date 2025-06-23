import torch
import torch.nn as nn
from typing import Optional, Dict
from einops import rearrange
from music_dit.utils import get_logger, get_hparams
from music_dit.modules.diffusion import DiT, DDIMSampler, TrainingLoss, Diffusion
from Code_for_Experiment.Targeted_Training.dit_training_on_vae.music_dit2.modules.embeddings.label_embedding import LabelEmbedding

logger = get_logger(__name__)

class MusicDiTModel(nn.Module):
    def __init__(self, *, input_dim: Optional[int] = None, hidden_dim: Optional[int] = None, context_dim: Optional[int] = None, num_layers: Optional[int] = None, num_heads: Optional[int] = None):
        super().__init__()
        hparams = get_hparams()
        device = hparams.device
        self.dit_model = DiT(
            input_dim=hparams.model.dit.input_dim if input_dim is None else input_dim,
            hidden_dim=hparams.model.dit.hidden_dim if hidden_dim is None else hidden_dim,
            num_layers=hparams.model.dit.num_layers if num_layers is None else num_layers,
            num_heads=hparams.model.dit.num_heads if num_heads is None else num_heads,
            dropout=hparams.model.dit.dropout,
            use_causality=hparams.model.dit.use_causality,
            use_cross_attention=hparams.model.dit.use_cross_attention,
            use_rpr=hparams.model.dit.use_rpr,
            context_dim=hparams.model.dit.context_dim if context_dim is None else context_dim,
            pos_embedding=hparams.model.dit.pos_embedding,
            max_position=hparams.model.dit.max_position,
            use_learned_variance=hparams.model.dit.use_learned_variance,
            max_timestep_period=hparams.model.dit.max_timestep_period,
            device=device
        )
        self.ddim_sampler = DDIMSampler(
            beta_start=hparams.model.sampler.beta_start,
            beta_end=hparams.model.sampler.beta_end,
            beta_schedule=hparams.model.sampler.beta_schedule,
            timestep_spacing=hparams.model.sampler.timestep_spacing,
            num_training_timesteps=hparams.model.sampler.num_training_timesteps,
            dynamic_thresholding_ratio=hparams.model.sampler.dynamic_thresholding_ratio,
            clip_sample_range=hparams.model.sampler.clip_sample_range,
            device=device
        )
        self.ddim_sampler.set_inference_timesteps(hparams.model.sampler.num_inference_timesteps)
        self.training_loss = TrainingLoss(
            sampler=self.ddim_sampler,
            loss_type=hparams.model.loss.loss_type
        )
        self.diffusion = Diffusion(
            model=self.dit_model,
            sampler=self.ddim_sampler,
            training_loss=self.training_loss
        )
        self.clap_embedding = nn.Linear(hparams.model.clap.embedding_dim, hparams.model.dit.hidden_dim, device=device)
        self.lyrics_embedding = LabelEmbedding(num_classes=hparams.model.lyrics.vocab_size, hidden_dim=hparams.model.dit.context_dim, device=device)
        self.input_dim = hparams.model.dit.input_dim
        self.hidden_dim = hparams.model.dit.hidden_dim
        self.clap_embedding_dim = hparams.model.clap.embedding_dim
        self.lyrics_vocab_size = hparams.model.lyrics.vocab_size
        self.vae_frame_size = hparams.model.vae.frame_size
        self.lyrics_padding_token_id = hparams.model.lyrics.padding_token

    @torch.no_grad()
    def preprocess_input(self, *, x: torch.Tensor, clap: torch.Tensor, prompt: torch.Tensor, lyrics: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        assert x.dim() == 4, "Input x must be 4-dimensional: (batch_size, num_frames, frame_size, input_dim)."
        assert x.size(2) == self.vae_frame_size, f"Frame size of input x must be {self.vae_frame_size}."
        assert x.size(-1) == self.input_dim, f"Dimension of input x must be {self.input_dim}."
        batch_size = x.size(0)
        num_frames = x.size(1)
        assert clap.dim() == 3, "clap must be 3-dimensional: (batch_size, num_frames, clap_embedding_dim)."
        assert clap.size() == (batch_size, num_frames, self.clap_embedding_dim), f"clap size must be (batch_size, num_frames, {self.clap_embedding_dim})."
        assert prompt.dim() == 4, "prompt must be 4-dimensional: (batch_size, prompt_length, frame_size, input_dim)."
        assert prompt.size() == (batch_size, prompt.size(1), self.vae_frame_size, self.input_dim), f"prompt size must be (batch_size, prompt_length, {self.vae_frame_size}, {self.input_dim})."
        assert lyrics.dim() == 2, "lyrics must be 2-dimensional: (batch_size, num_tokens)."
        assert lyrics.size(0) == batch_size, "lyrics batch size must be the same as vae_samples."
        if padding_mask is not None:
            assert padding_mask.dim() == 2, "padding_mask must be 2-dimensional: (batch_size, num_frames)."
            assert padding_mask.size() == (batch_size, num_frames), "padding_mask size must be (batch_size, num_frames)."
        lyrics_emb = self.lyrics_embedding(lyrics)
        clap_emb = self.clap_embedding(clap)
        x = rearrange(x, 'b f h d -> b (f h) d').contiguous()
        clap_emb = rearrange(clap_emb, 'b f d -> b f 1 d').expand(-1, -1, self.vae_frame_size, -1)
        clap_emb = rearrange(clap_emb, 'b f h d -> b (f h) d').contiguous()
        prompt = rearrange(prompt, 'b p h d -> b (p h) d').contiguous()
        if padding_mask is not None:
            padding_mask = rearrange(padding_mask, 'b f -> b f 1').expand(-1, -1, self.vae_frame_size)
            padding_mask = rearrange(padding_mask, 'b f h -> b (f h)').contiguous()
        return {
            "x": x,
            "prompt": prompt,
            "clap": clap_emb,
            "lyrics": lyrics_emb,
            "padding_mask": padding_mask,
        }

    def training_step(self, inp: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = self.diffusion.training_step(samples=inp["x"], prompt=inp["prompt"], conditions=inp["clap"], context=inp["lyrics"], padding_mask=inp["padding_mask"])
        return loss

    @torch.inference_mode()
    def inference(self, inp: Dict[str, torch.Tensor], *, start_timestep_index: int = 0, cfg_scale: Optional[float] = None, eta: float = 0.0) -> torch.Tensor:
        generated = self.diffusion.generate(x=inp["x"], prompt=inp["prompt"], conditions=inp["clap"], context=inp["lyrics"], start_timestep_index=start_timestep_index, cfg_scale=cfg_scale, eta=eta)
        generated = rearrange(generated, 'b (f h) d -> b f h d', h=self.vae_frame_size).contiguous()
        return generated