import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional
from ama_prof_divi.utils import merge_tensors
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.logging import get_logger
logger = get_logger(__name__)
import Mlp, MlpArgs
from ama_prof_divi.modules.transformers import (TransformerModelArgs, TransformerEncoder, Generator, InferAccelerationCache)
from ..chords_compressor import get_chords_compressor
from ..tokenizers import get_melody_tokenizer
from ..prompt_encoder import get_prompt_encoder

class MelodyGenerator(nn.Module):
    def __init__(self, hparams: dict):
        super(MelodyGenerator, self).__init__()
        self.hparams = hparams
        self.gen_hparams = self.hparams["ama-prof-divi"]["models"]["semantic"]["melody_generator"]
        self.device = self.hparams["ama-prof-divi"]["device"]
        self.tokenizer = get_melody_tokenizer(self.hparams)
        self.vocab_size = self.tokenizer.vocab_size
        self.feature_rate = self.gen_hparams["feature_rate"]
        self.window_size = self.gen_hparams["window_size_in_seconds"] * self.feature_rate
        self.generation_mode = self.gen_hparams["generation_mode"]
        self.temperature = self.gen_hparams["temperature"]
        self.top_p = self.gen_hparams["top_p"]
        assert self.window_size % 2 == 0 and self.window_size >= self.feature_rate * 2, \
            "Window size must be even.  Minimize window size is 2 seconds."
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
        chords_tokenizer = get_melody_tokenizer(self.hparams)
        chords_encoder_args = TransformerModelArgs(
            dim=self.gen_hparams["chords_encoder"]["dim"],
            num_layers=self.gen_hparams["chords_encoder"]["num_layers"],
            num_heads=self.gen_hparams["chords_encoder"]["num_heads"],
            dropout=self.gen_hparams["chords_encoder"]["dropout"],
            max_seq_len=self.gen_hparams["chords_encoder"]["max_seq_len"],
            hidden_dim=self.gen_hparams["chords_encoder"]["hidden_dim"],
            vocab_size=chords_tokenizer.vocab_size
        )
        self.chords_encoder = TransformerEncoder(chords_encoder_args, device=self.device)
        self.dim = self.prompt_encoder.joint_embedding_dim + self.chords_encoder.out_dim
        self.max_seq_len = self.window_size + 1
        max_positional_embeddings = chords_encoder_args.max_seq_len * self.max_seq_len + 1
        self.model_args = TransformerModelArgs(
            dim=self.dim,
            num_layers=self.gen_hparams["num_layers"],
            num_heads=self.gen_hparams["num_heads"],
            dropout=self.gen_hparams["dropout"],
            max_seq_len=self.max_seq_len,
            max_position_embeddings=max_positional_embeddings,
            hidden_dim=self.gen_hparams["hidden_dim"],
            vocab_size=self.vocab_size
        )
        self.generator = Generator(self.model_args,
                                   start_id=self.tokenizer.start_id,
                                   pad_id=self.tokenizer.pad_id,
                                   device=self.device)
        chords_compressor = get_chords_compressor(self.hparams)
        self.chords_compression_rate = chords_compressor.compress_ratio
        self.chords_window_size = self.window_size // self.chords_compression_rate
        logger.info("Melody generator is created.")
        logger.info("Melody generator dim = %d" % self.dim)
        logger.info("Melody vocab_size = %d" % self.vocab_size)
        logger.info("Melody generator window_size = %d (samples)" % self.window_size)
        logger.info("Melody feature_rate = %d (Hz)" % self.feature_rate)
        logger.info("Melody/chords compression rate = %d" % self.chords_compression_rate)
        logger.info("Generation mode = %s" % self.generation_mode)
        logger.info("Default temperature: %f" % self.temperature)
        logger.info("Default top_p: %f" % self.top_p)

    def _make_context(self,
                      text_prompt_embedding: torch.Tensor,
                      chord_sequence: [[int]]) -> (torch.Tensor, torch.Tensor):
        prompt_embedding = self.prompt_mlp(text_prompt_embedding.to(self.device))
        if type(chord_sequence) is list:
            max_len = max([len(c) for c in chord_sequence])
            chord_sequence_pt = torch.full((len(chord_sequence), max_len),
                                           self.tokenizer.pad_id,
                                           dtype=torch.long,
                                           device=self.device)
            for i in range(len(chord_sequence)):
                chord_sequence_pt[i, :len(chord_sequence[i])] = torch.tensor(chord_sequence[i],
                                                                             dtype=torch.long)
        elif type(chord_sequence) is torch.Tensor:
            chord_sequence_pt = chord_sequence.to(self.device)
        else:
            raise TypeError("chord_sequence must be either a list of lists or a torch.Tensor.")
        chords_embedding = self.chords_encoder(chord_sequence_pt)
        prompt_embedding = rearrange(prompt_embedding, "b d -> b () d")
        return prompt_embedding, chords_embedding

    def get_num_windows(self,
                        seq_len: int) -> int:
        half_window_size = self.window_size // 2
        num_windows = max(0, seq_len // half_window_size - 1)
        if seq_len % half_window_size != 0 or num_windows == 0:
            num_windows += 1
        return num_windows

    def prepare_for_training(self,
                             *,
                             text_prompt: [str],
                             chord_sequences: [[int]],
                             melody_sequences: [[int]],
                             start_window: int = 0,
                             num_windows: int = 1,
                             text_prompt_language: str = "en"):
        text_prompt_embedding, num_batches = self.prompt_encoder.get_text_prompt_embeddings(
            text_prompt=text_prompt,
            text_prompt_language=text_prompt_language)
        assert num_batches > 0, "Text prompt should not be empty."
        assert chord_sequences is not None
        assert len(chord_sequences) == num_batches, \
            f"Number of chord sequences {len(chord_sequences)} does not match the number of batches {num_batches}."
        assert melody_sequences is not None
        assert len(melody_sequences) == num_batches, \
            f"Number of melody sequences {len(melody_sequences)} does not match the number of batches {num_batches}."
        prompt_context, chord_context = self._make_context(text_prompt_embedding, chord_sequences)
        assert prompt_context.shape[0] == num_batches and chord_context.shape[0] == num_batches
        half_window_size = self.window_size // 2
        half_chords_window_size = self.chords_window_size // 2
        logger.info("window_size: %d" % self.window_size)
        logger.info("num_windows: %d" % num_windows)
        logger.info("prompt_context: %s" % str(prompt_context.shape))
        logger.info("chord_context: %s" % str(chord_context.shape))
        input_list = []
        for i in range(start_window, start_window + num_windows):
            chord_context_window = chord_context[:, i * half_chords_window_size:i * half_chords_window_size + self.chords_window_size, :]
            prompt_context_window = prompt_context.expand(-1, chord_context_window.shape[1], -1)
            context_window = torch.cat((prompt_context_window, chord_context_window), dim=-1)
            del chord_context_window
            del prompt_context_window
            if i == start_window:
                sentences_window = [sentence[:self.window_size + 1] for sentence in melody_sequences]
                prompt_len = 0
                add_start = True
            else:
                sentences_window = [sentence[i * half_window_size + 1:i * half_window_size + self.window_size] for sentence in melody_sequences]
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
                                                    i * half_chords_window_size,
                                                    dtype=torch.long)
            input_list.append(input_window)
        return {
            "tokens": merge_tensors([input_window["tokens"] for input_window in input_list]),
            "context": merge_tensors([input_window["context"] for input_window in input_list]),
            "labels": merge_tensors([input_window["labels"] for input_window in input_list]),
            "pos_bias": torch.cat([input_window["pos_bias"] for input_window in input_list]),
            "pos_bias_k": torch.cat([input_window["pos_bias_k"] for input_window in input_list])
        }

    @torch.inference_mode()
    def prepare_for_inference(self,
                              *,
                              chord_sequence: [[int]],
                              text_prompt: [str] = None,
                              text_prompt_language: str = "en",
                              text_prompt_embedding: torch.Tensor = None,
                              melody_prompt: [[int]] = None) -> dict:
        text_prompt_embedding, num_batches = self.prompt_encoder.get_text_prompt_embeddings(
            text_prompt=text_prompt,
            text_prompt_language=text_prompt_language,
            text_prompt_embedding=text_prompt_embedding)
        if num_batches == 0:
            num_batches = len(chord_sequence)
            text_prompt_embedding = text_prompt_embedding.expand(num_batches, -1)
        else:
            assert num_batches == len(chord_sequence), \
                "Batch size of chord_sequence does not match that of text_prompt."
        context = self._make_context(text_prompt_embedding, chord_sequence)
        prompt_context = context[0]
        chord_context = context[1]
        if melody_prompt is not None:
            assert len(melody_prompt) == num_batches, \
                "Batch size of melody_prompt does not match that of text_prompt."
            melody_prompt = [[self.tokenizer.start_id] + melody_prompt[i] for i in range(num_batches)]
        else:
            melody_prompt = [[self.tokenizer.start_id] for _ in range(num_batches)]
        return {
            "num_batches": num_batches,
            "prompt_context": prompt_context,
            "chord_context": chord_context,
            "melody_prompt": melody_prompt
        }

    @torch.inference_mode()
    def generate(self,
                 chord_sequence: [[int]],
                 *,
                 text_prompt: [str] = None,
                 text_prompt_language: str = "en",
                 text_prompt_embedding: torch.Tensor = None,
                 melody_prompt: [[int]] = None,
                 mode: str = None,
                 cache: InferAccelerationCache = None,
                 max_gen_len: int = -1,
                 temperature: float = -1.0,
                 top_p: float = -1.0) -> torch.Tensor:
        if temperature < 0.0:
            temperature = self.temperature
        if top_p < 0.0:
            top_p = self.top_p
        if mode is None:
            mode = self.generation_mode
        inputs = self.prepare_for_inference(
            chord_sequence=chord_sequence,
            text_prompt=text_prompt,
            text_prompt_language=text_prompt_language,
            text_prompt_embedding=text_prompt_embedding,
            melody_prompt=melody_prompt
        )
        logger.info("num_batches: %s" % inputs["num_batches"])
        logger.info("prompt_context: %s" % str(inputs["prompt_context"].shape))
        logger.info("chord_context: %s" % str(inputs["chord_context"].shape))
        logger.info("melody_prompt: %s" % str(inputs["melody_prompt"]))
        approx_seq_len = inputs["chord_context"].shape[1] * self.chords_compression_rate
        if 0 < max_gen_len < approx_seq_len:
            logger.warning("max_gen_len %d is smaller than the approx_seq_len %d.  ",
                           max_gen_len, approx_seq_len)
            approx_seq_len = max_gen_len
        half_window_size = self.window_size // 2
        half_chords_window_size = self.chords_window_size // 2
        num_windows = max(0, approx_seq_len // half_window_size - 1)
        if approx_seq_len % half_window_size != 0 or num_windows == 0:
            num_windows += 1
        logger.info("approx_seq_len: %d" % approx_seq_len)
        logger.info("window_size: %d" % self.window_size)
        logger.info("num_windows: %d" % num_windows)
        generated_sequences = torch.zeros((inputs["num_batches"], 0),
                                          dtype=torch.long).to("cpu")
        for i in range(num_windows):
            logger.info("Generating melody window %d/%d ..." % (i + 1, num_windows))
            chord_context = inputs["chord_context"][:, i * half_chords_window_size:i * half_chords_window_size + self.chords_window_size, :]
            prompt_context = inputs["prompt_context"].expand(-1, chord_context.shape[1], -1)
            context = torch.cat((prompt_context, chord_context), dim=2)
            logger.info("context: %s" % str(context.shape))
            if i == 0:
                proceeding_tokens = inputs["melody_prompt"]
                for p in proceeding_tokens:
                    assert p[0] == self.tokenizer.start_id, \
                        "The first token of the prompt must be '<|ss_start|>'."
            else:
                proceeding_tokens = generated_sequences[:, -half_window_size:].tolist()
            generated = self.generator.generate(
                prompt_tokens=proceeding_tokens,
                context=context,
                description="Generating melody window %d/%d ..." % (i + 1, num_windows),
                mode=mode,
                cache=cache,
                max_gen_len=self.window_size if i == 0 else half_window_size,
                temperature=temperature,
                top_p=top_p,
                pos_bias=i * half_window_size,
                pos_bias_k=i * half_chords_window_size)
            generated = torch.tensor(generated, dtype=torch.long)
            logger.info("generated: %s", generated.shape)
            generated_sequences = torch.cat((generated_sequences, generated), dim=1)
            if 0 < max_gen_len <= generated_sequences.shape[1]:
                generated_sequences = generated_sequences[:, :max_gen_len]
                logger.info("generated_sequences: %s" % str(generated_sequences.shape))
                break
            logger.info("generated_sequences: %s" % str(generated_sequences.shape))
        return generated_sequences

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
                         text_prompt: [str],
                         chord_sequence: [[int]],
                         melody_sequence: [[int]],
                         start_window: int = 0,
                         num_windows: int = 1,
                         text_prompt_language: str = "en") -> dict:
        inputs = self.prepare_for_training(text_prompt=text_prompt,
                                           chord_sequences=chord_sequence,
                                           melody_sequences=melody_sequence,
                                           start_window=start_window,
                                           num_windows=num_windows,
                                           text_prompt_language=text_prompt_language)
        logger.info("Executing forward pass for training...")
        logger.info("All tokens: %s" % str(inputs["tokens"].shape))
        logger.info("Context: %s" % str(inputs["context"].shape))
        logger.info("Labels: %s" % str(inputs["labels"].shape))
        return self.forward(all_tokens=inputs["tokens"],
                            context=inputs["context"],
                            labels=inputs["labels"],
                            pos_bias=inputs["pos_bias"],
                            pos_bias_k=inputs["pos_bias_k"])