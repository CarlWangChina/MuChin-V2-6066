import torch
import torch.nn as nn
from typing import Optional
from einops import rearrange
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.logging import get_logger
import get_logger
import Mlp, MlpArgs
from ama_prof_divi.modules.transformers import TransformerModelArgs, Generator, InferAccelerationCache
from ama_prof_divi.models.song_structures import SongStructureEncoder
from ..tokenizers import get_melody_tokenizer
import get_melody_tokenizer
import get_prompt_encoder
logger = get_logger(__name__)
DEFAULT_SONG_STRUCTURE = None

class ChordsGenerator(nn.Module):
    def __init__(self, hparams: dict):
        super(ChordsGenerator, self).__init__()
        self.hparams = hparams
        self.device = hparams["ama-prof-divi"]["device"]
        self.gen_hparams = self.hparams["ama-prof-divi"]["models"]["semantic"]["chords_generator"]
        self.tokenizer = get_melody_tokenizer(hparams)
        self.vocab_size = self.tokenizer.vocab_size
        logger.info("Chord vocab_size = %d" % self.vocab_size)
        self.prompt_encoder = get_prompt_encoder(hparams)
        logger.info("Joint embedding dim = %d" % self.prompt_encoder.joint_embedding_dim)
        prompt_mlp_args = self.gen_hparams["prompt_mlp"]
        mlp_args = MlpArgs(
            input_dim=self.prompt_encoder.joint_embedding_dim,
            output_dim=self.prompt_encoder.joint_embedding_dim,
            num_layers=prompt_mlp_args["num_layers"],
            activation=prompt_mlp_args["activation"]
        )
        if "dropout" in prompt_mlp_args:
            mlp_args.dropout = prompt_mlp_args["dropout"]
        if "hidden_dim" in prompt_mlp_args:
            mlp_args.hidden_dim = prompt_mlp_args["hidden_dim"]
        self.prompt_mlp = Mlp(mlp_args, device=self.device)
        self.song_structure_encoder = SongStructureEncoder(hparams)
        self.dim = self.prompt_encoder.joint_embedding_dim + self.song_structure_encoder.dim
        self.model_args = TransformerModelArgs(
            dim=self.dim,
            num_layers=self.gen_hparams["num_layers"],
            num_heads=self.gen_hparams["num_heads"],
            dropout=self.gen_hparams["dropout"],
            max_seq_len=self.gen_hparams["max_seq_len"],
            hidden_dim=self.gen_hparams["hidden_dim"],
            vocab_size=self.vocab_size
        )
        self.generator = Generator(self.model_args,
                                   start_id=self.tokenizer.start_id,
                                   pad_id=self.tokenizer.pad_id,
                                   end_id=self.tokenizer.end_id,
                                   device=self.device)
        self.generation_mode = self.gen_hparams["generation_mode"]
        self.temperature = self.gen_hparams["temperature"]
        self.top_p = self.gen_hparams["top_p"]

    def _make_context(self,
                      text_prompt_embedding: torch.Tensor,
                      lyrics: list,
                      num_batches: int = 0) -> torch.Tensor:
        prompt_embedding = self.prompt_mlp(text_prompt_embedding.to(self.device))
        lyrics_embedding = self.song_structure_encoder(lyrics)
        if lyrics_embedding.shape[0] == 1 and num_batches > 1:
            lyrics_embedding = lyrics_embedding.expand(num_batches, -1, -1)
        prompt_embedding = rearrange(prompt_embedding, "b d -> b () d")
        prompt_embedding = prompt_embedding.expand(-1, lyrics_embedding.shape[1], -1)
        context = torch.cat([prompt_embedding, lyrics_embedding], dim=-1)
        return context

    def prepare_for_training(self,
                             *,
                             text_prompt: list,
                             lyrics: list,
                             chord_sequences: list,
                             text_prompt_language: str = "en"):
        text_prompt_embedding, num_batches = self.prompt_encoder.get_text_prompt_embeddings(
            text_prompt=text_prompt,
            text_prompt_language=text_prompt_language
        )
        assert num_batches > 0, "Text prompt should not be empty."
        assert len(lyrics) == num_batches, \
            f"Number of lyrics {len(lyrics)} does not match the number of batches {num_batches}."
        assert len(chord_sequences) == num_batches, \
            f"Number of chord sequences {len(chord_sequences)} does not match the number of batches {num_batches}."
        context = self._make_context(text_prompt_embedding=text_prompt_embedding, lyrics=lyrics)
        assert context.shape[0] == num_batches
        return self.generator.prepare_for_autoregressive_training(sentences=chord_sequences,
                                                                  contexts=context,
                                                                  add_start_id_at_beginning=True,
                                                                  device=self.device)

    @torch.inference_mode()
    def prepare_for_inference(self,
                              *,
                              text_prompt: list = None,
                              text_prompt_language: str = "en",
                              text_prompt_embedding: torch.Tensor = None,
                              lyrics: list = None,
                              chord_prompt: list = None) -> dict:
        text_prompt_embedding, num_batches = self.prompt_encoder.get_text_prompt_embeddings(
            text_prompt=text_prompt,
            text_prompt_language=text_prompt_language,
            text_prompt_embedding=text_prompt_embedding
        )
        text_prompt_embedding.to(self.device)
        if lyrics is None:
            lyrics = [DEFAULT_SONG_STRUCTURE]
        else:
            if num_batches == 0:
                num_batches = len(lyrics)
            text_prompt_embedding = text_prompt_embedding.expand(num_batches, -1)
            assert len(lyrics) == num_batches, \
                f"Number of lyrics {len(lyrics)} does not match the number of text prompts {num_batches}."
        context = self._make_context(num_batches=num_batches,
                                     text_prompt_embedding=text_prompt_embedding,
                                     lyrics=lyrics)
        if chord_prompt is None:
            if num_batches == 0:
                num_batches = 1
            chord_prompt = [[self.tokenizer.start_id] for _ in range(num_batches)]
        else:
            chord_prompt = [[self.tokenizer.start_id] + chord for chord in chord_prompt]
            if num_batches != 0:
                assert num_batches == len(chord_prompt), \
                    (f"Number of batches {num_batches} does not match the number of chord prompts "
                     f"{len(chord_prompt)}.")
            else:
                num_batches = len(chord_prompt)
            context = context.expand(num_batches, -1, -1)
        return {
            "num_batches": num_batches,
            "context": context,
            "chord_prompt": chord_prompt
        }

    def forward(self,
                all_tokens: torch.Tensor,
                context: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                cache: Optional[InferAccelerationCache] = None,
                start_pos: int = 0,
                pos_bias: int = 0,
                pos_bias_k: int = 0) -> dict:
        return self.generator.forward(all_tokens,
                                      context=context,
                                      labels=labels,
                                      cache=cache,
                                      start_pos=start_pos,
                                      pos_bias=pos_bias,
                                      pos_bias_k=pos_bias_k)

    def forward_training(self,
                         *,
                         text_prompt: list,
                         lyrics: list,
                         chord_sequences: list,
                         text_prompt_language: str = "en"):
        inputs = self.prepare_for_training(text_prompt=text_prompt,
                                           lyrics=lyrics,
                                           chord_sequences=chord_sequences,
                                           text_prompt_language=text_prompt_language)
        logger.info("Executing forward pass for training...")
        logger.info("All tokens: %s" % str(inputs["tokens"].shape))
        logger.info("Context: %s" % str(inputs["context"].shape))
        logger.info("Labels: %s" % str(inputs["labels"].shape))
        return self.forward(all_tokens=inputs["tokens"],
                            context=inputs["context"],
                            labels=inputs["labels"])

    @torch.inference_mode()
    def generate(self,
                 *,
                 text_prompt: list = None,
                 text_prompt_language: str = "en",
                 text_prompt_embedding: torch.Tensor = None,
                 chord_prompt: list = None,
                 lyrics: list = None,
                 cache: InferAccelerationCache = None,
                 mode: str = None,
                 max_gen_len: int = -1,
                 temperature: float = -1.0,
                 top_p: float = -1.0) -> list:
        if temperature < 0.0:
            temperature = self.temperature
        if top_p < 0.0:
            top_p = self.top_p
        if mode is None:
            mode = self.generation_mode
        inputs = self.prepare_for_inference(text_prompt=text_prompt,
                                            text_prompt_language=text_prompt_language,
                                            text_prompt_embedding=text_prompt_embedding,
                                            lyrics=lyrics,
                                            chord_prompt=chord_prompt)
        for p in inputs["chord_prompt"]:
            assert p[0] == self.tokenizer.start_id, \
                "The first token of the prompt must be '<|ss_start|>'."
        return self.generator.generate(prompt_tokens=inputs["chord_prompt"],
                                       context=inputs["context"],
                                       description="Generating chords...",
                                       mode=mode,
                                       cache=cache,
                                       max_gen_len=max_gen_len,
                                       temperature=temperature,
                                       top_p=top_p)