import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange
from ama_prof_divi.modules.transformers import TransformerModelArgs, Generator, TransformerEncoder, InferAccelerationCache
from ama_prof_divi.modules.diffusion import LatentDiffusion, DiffusionArgs, UnetModelArgs, WaveNetModelArgs
from ama_prof_divi.utils import merge_tensors, sample_top_p, safe_softmax
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.logging import get_logger
logger = get_logger(__name__)
from ama_prof_divi.models.song_structures import get_ss_tokenizer
from ..encodec import get_encodec_wrapper
from . import get_melody_tokenizer

class AccompanimentGenerator(nn.Module):
    def __init__(self, hparams: dict, enable_diffusion: bool or str = "auto", device: Optional[torch.device or str] = None):
        super(AccompanimentGenerator, self).__init__()
        self.hparams = hparams
        self.encodec = get_encodec_wrapper(hparams)
        if device is None:
            device = hparams["ama-prof-divi"]["device"]
        transformer_args = hparams["ama-prof-divi"]["models"]["acoustic"]["transformer"]
        diffusion_args = hparams["ama-prof-divi"]["models"]["acoustic"]["diffusion"]
        context_encoder_args = hparams["ama-prof-divi"]["models"]["acoustic"]["context_encoder"]
        self.generation_mode = transformer_args["generation_mode"]
        self.temperature = transformer_args["temperature"]
        self.top_p = transformer_args["top_p"]
        windows_size_in_seconds = transformer_args["window_size_in_seconds"]
        assert windows_size_in_seconds > 0 and windows_size_in_seconds % 2 == 0, \
            f"windows_size_in_seconds ({windows_size_in_seconds}) should be a positive even number."
        self.vocab_size = self.encodec.codebook_size
        self.start_id = self.vocab_size + 1
        self.pad_id = self.vocab_size + 2
        self.vocab_size += 4
        self.dim = diffusion_args["model_channels"]
        self.transformer_args = TransformerModelArgs(
            dim=self.dim,
            num_layers=transformer_args["num_layers"],
            num_heads=transformer_args["num_heads"],
            dropout=transformer_args["dropout"],
            max_seq_len=transformer_args["max_seq_len"],
            vocab_size=self.vocab_size
        )
        self.generator = Generator(self.transformer_args,
                                   start_id=self.start_id,
                                   pad_id=self.pad_id,
                                   end_id=-1,
                                   device=device)
        self.semantic_rvq_tokenizer = get_melody_tokenizer(hparams)
        self.semantic_rvq_vocab_size = self.semantic_rvq_tokenizer.vocab_size
        self.semantic_rvq_num_q = context_encoder_args["num_quantizers"]
        self.semantic_rvq_pad_id = self.semantic_rvq_tokenizer.pad_id
        self.context_encoder_args = TransformerModelArgs(
            dim=self.transformer_args.dim,
            num_layers=context_encoder_args["num_layers"],
            num_heads=context_encoder_args["num_heads"],
            dropout=context_encoder_args["dropout"],
            num_quantization_groups=self.semantic_rvq_num_q,
            max_seq_len=self.transformer_args.max_seq_len,
            vocab_size=self.semantic_rvq_vocab_size,
        )
        self.context_encoder = TransformerEncoder(self.context_encoder_args,
                                                  device=device)
        self.melody_pad_id = context_encoder_args["melody_pad_id"]
        self.melody_vocab_size = context_encoder_args["melody_vocab_size"]
        self.melody_embedding = nn.Embedding(self.melody_vocab_size,
                                             self.dim,
                                             padding_idx=self.melody_pad_id,
                                             device=device)
        self.ss_tokenizer = get_ss_tokenizer(hparams)
        self.ss_vocab_size = self.ss_tokenizer.vocab_size
        self.ss_pad_id = self.ss_tokenizer.pad_id
        self.ss_embedding = nn.Embedding(self.ss_vocab_size,
                                         self.dim,
                                         padding_idx=self.ss_pad_id,
                                         device=device)
        self.prompt_embedding_dim = diffusion_args["prompt_dim"]
        self.num_quantizers = self.encodec.num_quantizers
        melody_feature_rate = hparams["ama-prof-divi"]["models"]["semantic"]["encoder"]["features_rate"]
        self.acoustic_feature_rate = self.encodec.frame_rate * self.num_quantizers
        self.acoustic_melody_feature_ratio = self.acoustic_feature_rate / melody_feature_rate
        self.window_size = windows_size_in_seconds * self.acoustic_feature_rate
        self.melody_window_size = windows_size_in_seconds * melody_feature_rate
        logger.info("num_quantizers: %d", self.num_quantizers)
        logger.info("acoustic_feature_rate: %d", self.acoustic_feature_rate)
        logger.info("melody_feature_rate: %d", melody_feature_rate)
        logger.info("window_size: %d", self.window_size)
        logger.info("melody_window_size: %d", self.melody_window_size)
        logger.info("acoustic_melody_feature_ratio: %f", self.acoustic_melody_feature_ratio)
        if enable_diffusion == "auto":
            enable_diffusion = diffusion_args["enabled"]
        self.denoiser = None
        if enable_diffusion:
            self.denoiser = diffusion_args["denoiser"]
            if self.denoiser == "unet":
                unet_args = UnetModelArgs(
                    in_channels=self.dim,
                    out_channels=self.dim,
                    model_channels=self.dim,
                    context_dim=diffusion_args["prompt_dim"],
                    num_res_blocks=diffusion_args["unet"]["num_res_blocks"],
                    attention_resolutions=diffusion_args["unet"]["attention_resolutions"],
                    dropout=diffusion_args["unet"]["dropout"],
                    channel_mult=diffusion_args["unet"]["channel_mult"],
                    conv_resample=diffusion_args["unet"]["conv_resample"],
                    dims=1,
                    num_heads=diffusion_args["unet"]["num_heads"],
                    use_transformer=diffusion_args["unet"]["use_transformer"],
                    transformer_depth=diffusion_args["unet"]["transformer_depth"],
                    use_scale_shift_norm=diffusion_args["unet"]["use_scale_shift_norm"],
                    res_block_updown=diffusion_args["unet"]["res_block_updown"],
                    use_time_embedding=diffusion_args["unet"]["use_time_embedding"],
                    use_controlnet=False
                )
                wavenet_args = None
            elif self.denoiser == "wavenet":
                unet_args = None
                wavenet_args = WaveNetModelArgs(
                    in_channels=self.dim,
                    out_channels=self.dim,
                    model_channels=self.dim,
                    context_channels=diffusion_args["prompt_dim"],
                    num_layers=diffusion_args["wavenet"]["num_layers"],
                    dilation_cycle=diffusion_args["wavenet"]["dilation_cycle"],
                )
            else:
                raise ValueError("Invalid diffusion denoiser '%s'." % diffusion_args["model"])
            diffusion_args = DiffusionArgs(
                sampler=diffusion_args["sampler"]["name"],
                sampler_extra_args=diffusion_args["sampler"],
                denoiser=diffusion_args["denoiser"],
                unet=unet_args,
                wavenet=wavenet_args,
                guidance_scale=diffusion_args["guidance_scale"]
            )
            del diffusion_args.sampler_extra_args["name"]
            self.diffusion_embedding = nn.Embedding(self.vocab_size,
                                                    self.dim,
                                                    padding_idx=self.pad_id,
                                                    device=device)
            self.diffusion = LatentDiffusion(diffusion_args,
                                             device=device)
            self.diffusion_for_training = LatentDiffusion(diffusion_args,
                                                          training=True,
                                                          device=device)
            self.diffusion_norm = nn.LayerNorm(self.dim,
                                               device=device)
            self.diffusion_output_proj = nn.Linear(self.dim,
                                                   self.vocab_size,
                                                   device=device)
        else:
            self.diffusion = None

    @property
    def device(self):
        return self.encodec.device

    def _get_context_embeddings(self, *, rvq_tokens: torch.Tensor, melody_tokens: Optional[torch.Tensor], ss_tokens: Optional[torch.Tensor]) -> torch.Tensor:
        assert rvq_tokens.dim() == 3, \
            f"rvq_tokens ({rvq_tokens.shape}) should be 3D."
        assert rvq_tokens.shape[2] == self.context_encoder_args.num_quantization_groups, \
            (f"rvq_tokens.shape[2] ({rvq_tokens.shape[2]}) should be equal to "
             f"self.context_encoder_args.num_quantization_groups "
             f"({self.context_encoder_args.num_quantization_groups}).")
        emb = None
        if melody_tokens is not None:
            assert melody_tokens.dim() == 2, \
                f"melody_tokens ({melody_tokens.shape}) should be 2D."
            assert melody_tokens.shape[0] == rvq_tokens.shape[0]
            assert melody_tokens.shape[1] == rvq_tokens.shape[1]
            emb = self.melody_embedding(melody_tokens)
        if ss_tokens is not None:
            assert ss_tokens.dim() == 2, \
                f"ss_tokens ({ss_tokens.shape}) should be 2D."
            assert ss_tokens.shape[0] == rvq_tokens.shape[0]
            assert ss_tokens.shape[1] == rvq_tokens.shape[1]
            ss_emb = self.ss_embedding(ss_tokens)
            if emb is not None:
                emb += ss_emb
            else:
                emb = ss_emb
        return self.context_encoder(rvq_tokens, emb=emb)

    @torch.inference_mode()
    def prepare_for_inference(self, *, rvq_tokens: torch.Tensor, melody_tokens: Optional[torch.Tensor], ss_tokens: Optional[torch.Tensor], prompt: Optional[torch.Tensor]) -> dict:
        context = self._get_context_embeddings(rvq_tokens=rvq_tokens,
                                               melody_tokens=melody_tokens,
                                               ss_tokens=ss_tokens)
        if prompt is not None:
            assert prompt.dim() == 2 or prompt.dim() == 3, f"prompt ({prompt.shape}) should be 2D or 3D."
            assert prompt.shape[0] == context.shape[0], \
                f"prompt.shape[0] ({prompt.shape[0]}) should be equal to context.shape[0] ({context.shape[0]})."
            assert prompt.shape[-1] == self.prompt_embedding_dim, \
                f"prompt.shape[-1] ({prompt.shape[1]}) should be equal to {self.prompt_embedding_dim}."
            if prompt.dim() == 2:
                prompt = rearrange(prompt, 'b d -> b d 1')
        return {
            "num_batches": context.shape[0],
            "prompt_embedding": prompt,
            "context": context
        }

    @torch.inference_mode()
    def generate(self, *, rvq_tokens: torch.Tensor, melody_tokens: Optional[torch.Tensor], ss_tokens: Optional[torch.Tensor], prompt: Optional[torch.Tensor], cache: InferAccelerationCache = None, mode: str = None, max_gen_len: int = -1, temperature: float = -1.0, top_p: float = -1.0) -> torch.Tensor:
        if temperature < 0.0:
            temperature = self.temperature
        if top_p < 0.0:
            top_p = self.top_p
        if mode is None:
            mode = self.generation_mode
        inputs = self.prepare_for_inference(rvq_tokens=rvq_tokens,
                                            melody_tokens=melody_tokens,
                                            ss_tokens=ss_tokens,
                                            prompt=prompt)
        approx_seq_len = int(inputs["context"].shape[1] * self.acoustic_melody_feature_ratio)
        if 0 < max_gen_len < approx_seq_len:
            logger.warning("max_gen_len %d is smaller than the approx_seq_len %d.  ",
                           max_gen_len, approx_seq_len)
            approx_seq_len = max_gen_len
        assert approx_seq_len < self.transformer_args.max_seq_len, \
            (f"approx_seq_len ({approx_seq_len}) should be smaller than the maximum sequence length "
             f"({self.transformer_args.max_seq_len}).")
        half_window_size = self.window_size // 2
        half_melody_window_size = self.melody_window_size // 2
        num_windows = max(0, approx_seq_len // half_window_size - 1)
        if approx_seq_len % half_window_size != 0 or num_windows == 0:
            num_windows += 1
        logger.info("window_size: %d, melody_window_size: %d", self.window_size, self.melody_window_size)
        logger.info("approx_seq_len: %d, num_windows: %d", approx_seq_len, num_windows)
        generated_sequences = torch.zeros((inputs["num_batches"], 0),
                                          dtype=torch.long).to("cpu")
        for i in range(num_windows):
            logger.info("Generating acoustic window %d/%d ..." % (i + 1, num_windows))
            context = inputs["context"][:, i * half_melody_window_size:i * half_melody_window_size + self.melody_window_size, :]
            if i == 0:
                proceeding_tokens = [[self.start_id]] * inputs["num_batches"]
            else:
                proceeding_tokens = generated_sequences[:, -half_window_size:].tolist()
            generated = self.generator.generate(
                prompt_tokens=proceeding_tokens,
                context=context,
                description="Generating acoustic window %d/%d ..." % (i + 1, num_windows),
                mode=mode,
                cache=cache,
                max_gen_len=self.window_size if i == 0 else half_window_size,
                temperature=temperature,
                top_p=top_p,
                pos_bias=i * half_window_size,
                pos_bias_k=i * half_melody_window_size,
                output="tensor")
            if self.diffusion is not None:
                generated = self.diffusion_embedding(generated)
                generated = rearrange(generated, "b l d -> b d l")
                generated = self._generate_tokens_by_diffusion(x_start=generated,
                                                               context=inputs["prompt_embedding"],
                                                               mode=mode,
                                                               temperature=temperature,
                                                               top_p=top_p,
                                                               description="Refining acoustic window %d/%d ..." %
                                                                           (i + 1, num_windows))
            generated_sequences = torch.cat((generated_sequences,
                                             generated.to("cpu")), dim=1)
            if 0 < max_gen_len <= generated_sequences.shape[1]:
                generated_sequences = generated_sequences[:, :max_gen_len]
                logger.info("generated acoustic sequences: %s" % str(generated_sequences.shape))
                break
        logger.info("generated acoustic sequences: %s" % str(generated_sequences.shape))
        return generated_sequences

    @property
    def controlnet_enabled(self):
        return self.diffusion.model.controlnet is not None

    def _generate_tokens_by_diffusion(self,
                                      x_start: torch.Tensor,
                                      context: torch.Tensor,
                                      mode: str = "sample_top_p",
                                      temperature: float = -1.0,
                                      top_p: float = -1.0,
                                      description: str = None) -> torch.Tensor:
        assert self.diffusion is not None, "Diffusion has not been initialized."
        logits = self.diffusion.generate(context=context,
                                         seq_len=x_start.shape[2],
                                         latent_start=x_start,
                                         description=description)
        logits = rearrange(logits, "b d l -> b l d")
        logits = self.diffusion_norm(logits)
        logits = self.diffusion_output_proj(logits)
        if mode == "sample_top_p" and temperature > 0:
            probs = safe_softmax(logits / temperature, dim=-1)
            tokens = sample_top_p(probs, top_p)
        elif mode == "greedy" or temperature <= 0:
            tokens = torch.argmax(logits, dim=-1)
        else:
            raise ValueError("Invalid generation mode '%s'." % mode)
        return tokens

    def train_transformer(self, train_mode: bool = True):
        assert self.generator is not None, "Generator has not been initialized."
        assert self.diffusion is None, "When training the transformer, diffusion should be disabled."
        self.train(train_mode)
        logger.info("Acoustic generator: Set to transformer %s mode.",
                    "training" if train_mode else "eval")

    def train_diffusion(self, train_mode: bool = True):
        assert self.generator is not None, "Generator has not been initialized."
        assert self.diffusion is not None, "When training the diffusion, diffusion should be enabled."
        self.train(train_mode)
        logger.info("Acoustic generator: Set to diffusion %s mode.", "training" if train_mode else "eval")
        if train_mode:
            for p in self.generator.parameters():
                p.requires_grad = False
            for p in self.melody_embedding.parameters():
                p.requires_grad = False
            logger.info("Acoustic generator: The parameters of the transformer have been locked.")

    def train_controlnet(self, train_mode: bool = True):
        self.train_diffusion(train_mode=train_mode)
        self.diffusion.freeze_unet_model(freeze=not train_mode)
        logger.info("Acoustic generator: Set to control net %s mode.", "training" if train_mode else "eval")

    def train_output_projector(self, train_mode: bool = True):
        assert self.generator is not None, "Generator has not been initialized."
        self.train(train_mode)
        if train_mode:
            for p in self.generator.parameters():
                p.requires_grad = False
        logger.info("Acoustic generator: Set to output projector %s mode.",
                    "training" if train_mode else "eval")

    def get_num_windows(self, seq_len: int) -> int:
        half_window_size = self.window_size // 2
        num_windows = max(0, seq_len // half_window_size - 1)
        if seq_len % half_window_size != 0 or num_windows == 0:
            num_windows += 1
        return num_windows

    def prepare_for_training(self, *, rvq_tokens: torch.Tensor, melody_tokens: Optional[torch.Tensor], ss_tokens: Optional[torch.Tensor], acoustic_tokens: torch.Tensor, start_window: int = 0, num_windows: int = 1) -> dict:
        context = self._get_context_embeddings(rvq_tokens=rvq_tokens,
                                               melody_tokens=melody_tokens,
                                               ss_tokens=ss_tokens)
        assert acoustic_tokens.dim() == 2, \
            f"acoustic_tokens ({acoustic_tokens.shape}) should be 2D."
        assert acoustic_tokens.shape[0] == context.shape[0], \
            (f"acoustic_tokens.shape[0] ({acoustic_tokens.shape[0]}) should be equal to context.shape[0] "
             f"({context.shape[0]}).")
        half_window_size = self.window_size // 2
        half_melody_window_size = self.melody_window_size // 2
        input_list = []
        for i in range(start_window, start_window + num_windows):
            context_window = context[:, i * half_melody_window_size:i * half_melody_window_size + self.melody_window_size, :]
            if i == start_window:
                sentences_window = [sentence[:self.window_size + 1] for sentence in acoustic_tokens]
                prompt_len = 0
                add_start = True
            else:
                sentences_window = [sentence[i * half_window_size + 1:i * half_window_size + self.window_size] for sentence in acoustic_tokens]
                prompt_len = len(sentences_window[0]) // 2
                add_start = False
            input_window = self.generator.prepare_for_autoregressive_training(sentences=sentences_window,
                                                                              contexts=context_window,
                                                                              prompt_len=prompt_len,
                                                                              add_start_id_at_beginning=add_start,
                                                                              device=self.device)
            mini_batch_size = len(input_window["tokens"])
            input_window["pos_bias"] = torch.full((mini_batch_size,),
                                                  half_window_size, dtype=torch.long)
            input_window["pos_bias_k"] = torch.full((mini_batch_size,),
                                                    i * half_melody_window_size,
                                                    dtype=torch.long)
            input_list.append(input_window)
        return {
            "tokens": merge_tensors([input_window["tokens"] for input_window in input_list]),
            "context": merge_tensors([input_window["context"] for input_window in input_list]),
            "labels": merge_tensors([input_window["labels"] for input_window in input_list]),
            "pos_bias": torch.cat([input_window["pos_bias"] for input_window in input_list]),
            "pos_bias_k": torch.cat([input_window["pos_bias_k"] for input_window in input_list])
        }

    def forward_training_transformer(self, *, rvq_tokens: torch.Tensor, melody_tokens: Optional[torch.Tensor], ss_tokens: Optional[torch.Tensor], acoustic_tokens: torch.Tensor, start_window: int = 0, num_windows: int = 1) -> dict:
        inputs = self.prepare_for_training(rvq_tokens=rvq_tokens,
                                           melody_tokens=melody_tokens,
                                           ss_tokens=ss_tokens,
                                           acoustic_tokens=acoustic_tokens,
                                           start_window=start_window,
                                           num_windows=num_windows)
        return self.generator(all_tokens=inputs["tokens"],
                              context=inputs["context"],
                              labels=inputs["labels"],
                              pos_bias=inputs["pos_bias"],
                              pos_bias_k=inputs["pos_bias_k"])

    def _validate_diffusion_input(self, *, acoustic_tokens: torch.Tensor, prompt_embedding: Optional[torch.Tensor]) -> None:
        assert self.diffusion is not None, "Diffusion is not enabled."
        assert acoustic_tokens.dim() == 2, \
            f"acoustic_tokens ({acoustic_tokens.shape}) should be 2D."
        if prompt_embedding is not None:
            assert prompt_embedding.shape == (acoustic_tokens.shape[0], self.prompt_embedding_dim), \
                (f"prompt_embedding.shape ({prompt_embedding.shape}) should be equal to "
                 f"(num_batches, {self.prompt_embedding_dim}).")

    def forward_training_diffusion(self, *, acoustic_tokens: torch.Tensor, prompt_embedding: Optional[torch.Tensor]) -> float:
        self._validate_diffusion_input(acoustic_tokens=acoustic_tokens,
                                       prompt_embedding=prompt_embedding)
        latent = self.diffusion_embedding(acoustic_tokens.to(self.device))
        latent = rearrange(latent, "b l d -> b d l")
        result = self.diffusion_for_training(latent=latent,
                                             context=prompt_embedding)
        return result["loss"].item()

    def forward_training_controlnet(self, *, acoustic_tokens: torch.Tensor, prompt_embedding: Optional[torch.Tensor]) -> float:
        self._validate_diffusion_input(acoustic_tokens=acoustic_tokens,
                                       prompt_embedding=prompt_embedding)
        condition = self.diffusion_embedding(acoustic_tokens.to(self.device))
        condition = rearrange(condition, "b l d -> b d l")
        condition = F.dropout(condition, p=0.2)
        latent = self.diffusion_embedding(acoustic_tokens.to(self.device))
        latent = rearrange(latent, "b l d -> b d l")
        result = self.diffusion_for_training(latent=latent,
                                             condition=condition,
                                             context=prompt_embedding)
        return result["loss"].item()

    def forward_training_output(self, *, acoustic_tokens: torch.Tensor) -> float:
        assert acoustic_tokens.dim() == 2, \
            f"acoustic_tokens ({acoustic_tokens.shape}) should be 2D."
        logits = self.diffusion_embedding(acoustic_tokens.to(self.device))
        logits = self.diffusion_norm(logits)
        logits = self.diffusion_output_proj(logits)
        labels = acoustic_tokens.to(self.device)
        loss = F.cross_entropy(logits.transpose(1, 2),
                               labels,
                               ignore_index=self.pad_id)
        return loss.item()