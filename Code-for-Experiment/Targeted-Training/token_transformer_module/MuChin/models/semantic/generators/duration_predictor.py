import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from tqdm import tqdm
from ama_prof_divi.utils import init_embedding, sample_top_p, safe_softmax
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
from ama_prof_divi.modules.transformers import (TransformerEncoder, TransformerDecoder, TransformerModelArgs, InferAccelerationCache, prepare_for_autoregressive_training)
from ama_prof_divi.models.lyrics import get_phoneme_tokenizer
from ama_prof_divi.models.song_structures import get_ss_tokenizer
logger = get_logger(__name__)

class DPEncoder(nn.Module):
    def __init__(self, dim: int, num_layers: int, num_heads: int, max_seq_len: int, dropout: float = 0.0):
        super(DPEncoder, self).__init__()
        self.phoneme_tokenizer = get_phoneme_tokenizer()
        self.ss_tokenizer = get_ss_tokenizer()
        self.dim = dim
        self.ss_embedding = nn.Embedding(self.ss_tokenizer.vocab_size, dim)
        init_embedding_(self.ss_embedding)
        args = TransformerModelArgs(dim=self.dim, num_layers=num_layers, num_heads=num_heads, max_seq_len=max_seq_len, vocab_size=self.phoneme_tokenizer.vocab_size, dropout=dropout)
        self.model = TransformerEncoder(args)
        logger.info("Duration predictor encoder created.")

    @property
    def device(self):
        return self.model.device

    def forward(self, phonemes: torch.Tensor, *, ss: torch.Tensor, prompt_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert phonemes.shape == ss.shape, "Shape of phonemes and ss should be the same."
        assert phonemes.dim() == 2, "The input tensors should be 2D, in shape (num_batches, seq_len)."
        assert torch.min(phonemes) >= 0 and torch.max(phonemes) < self.phoneme_tokenizer.vocab_size, f"Invalid value of phonemes.  The value should be in the range of [0, {self.phoneme_tokenizer.vocab_size})."
        assert torch.min(ss) >= 0 and torch.max(ss) < self.ss_tokenizer.vocab_size, f"Invalid value of ss.  The value should be in the range of [0, {self.ss_tokenizer.vocab_size})."
        num_batches, seq_len = phonemes.shape
        if prompt_embedding is not None:
            assert prompt_embedding.shape == (num_batches, self.dim), ("The prompt embedding should be 2D, in shape (num_batches, dim).")
        return self.model(phonemes, emb=self.ss_embedding(ss) if prompt_embedding is None else prompt_embedding + self.ss_embedding(ss))

class DPDecoder(nn.Module):
    def __init__(self, dim: int, num_layers: int, num_heads: int, max_seq_len: int, max_duration: int, dropout: float = 0.0):
        super(DPDecoder, self).__init__()
        self.start_id = max_duration
        self.end_id = max_duration + 1
        self.pad_id = max_duration + 2
        self.sep_id = max_duration + 3
        self.vocab_size = max_duration + 4
        self.ss_tokenizer = get_ss_tokenizer()
        self.dim = dim
        self.pause_start_id = 2
        self.pause_pad_id = 3
        self.pause_end_id = 4
        self.pause_vocab_size = 5
        self.pause_embedding = nn.Embedding(self.pause_vocab_size, self.dim)
        init_embedding_(self.pause_embedding)
        self.ss_embedding = nn.Embedding(self.ss_tokenizer.vocab_size, dim)
        init_embedding_(self.ss_embedding)
        self.args = TransformerModelArgs(dim=self.dim, num_layers=num_layers, num_heads=num_heads, max_seq_len=max_seq_len, vocab_size=self.vocab_size, dropout=dropout)
        self.model = TransformerDecoder(self.args, output_dim=self.dim)
        self.ln = nn.LayerNorm(self.dim)
        self.output_proj = nn.Linear(self.dim, 1)
        self.output_pause_proj = nn.Linear(self.dim, self.pause_vocab_size)
        self.output_ss_proj = nn.Linear(self.dim, self.ss_tokenizer.vocab_size)
        logger.info("Duration predictor decoder created.")

    @property
    def device(self):
        return self.model.device

    def forward(self, duration: torch.Tensor, *, pause: torch.Tensor, ss: torch.Tensor, context: torch.Tensor, cache: Optional[InferAccelerationCache] = None, start_pos: int = 0) -> dict:
        assert duration.dim() == 2, "The input tensors should be 2D."
        assert duration.shape == pause.shape == ss.shape, "Shape of duration, pause, and ss should be the same."
        assert context.dim() == 3, "The context tensor should be 3D."
        assert context.shape[0] == duration.shape[0], "The batch size of context should be the same as duration."
        assert context.shape[-1] == self.dim, ("The last dimension of context should be the same as the hidden dimension.")
        if duration.dtype != torch.long:
            duration = duration.long()
        if pause.dtype != torch.long:
            pause = pause.long()
        if ss.dtype != torch.long:
            ss = ss.long()
        assert torch.min(duration) >= 0 and torch.max(duration) <= self.vocab_size, "Invalid duration value."
        assert torch.min(pause) >= 0 and torch.max(pause) <= 5, "Invalid pause value."
        assert torch.min(ss) >= 0 and torch.max(ss) < self.ss_tokenizer.vocab_size, "Invalid song structure value."
        logits = self.model(duration, emb=self.ss_embedding(ss) + self.pause_embedding(pause), context=context, cache=cache, start_pos=start_pos)
        logits = self.ln(logits)
        duration_logits = self.output_proj(logits).squeeze(-1)
        duration_logits = torch.clamp(duration_logits, min=0, max=self.vocab_size - 1)
        pause_logits = self.output_pause_proj(logits)
        ss_logits = self.output_ss_proj(logits)
        return {
            "duration_logits": duration_logits,
            "pause_logits": pause_logits,
            "ss_logits": ss_logits
        }

class DurationPredictor(nn.Module):
    def __init__(self, hparams: dict):
        super(DurationPredictor, self).__init__()
        self.hparams = hparams
        self.dp_hparams = hparams["ama_prof_divi"]["models"]["semantic"]["duration_predictor"]
        device = hparams["ama_prof_divi"]["device"]
        self.time_unit_hz = self.dp_hparams["time_unit_hz"]
        self.max_seq_len = self.dp_hparams["max_seq_len"]
        self.ss_tokenizer = get_ss_tokenizer()
        self.dim = self.dp_hparams["hidden_dim"]
        self.max_duration = self.dp_hparams["max_duration"]
        self.encoder = DPEncoder(dim=self.dim, num_layers=self.dp_hparams["encoder"]["num_layers"], num_heads=self.dp_hparams["encoder"]["num_heads"], max_seq_len=self.max_seq_len, dropout=self.dp_hparams["encoder"]["dropout"]).to(device)
        self.decoder = DPDecoder(dim=self.dim, num_layers=self.dp_hparams["decoder"]["num_layers"], num_heads=self.dp_hparams["decoder"]["num_heads"], max_seq_len=self.max_seq_len, max_duration=self.max_duration, dropout=self.dp_hparams["decoder"]["dropout"]).to(device)
        self.default_seq_len_ratio = self.dp_hparams["default_seq_len_ratio"]
        self.generation_mode = self.dp_hparams["decoder"]["generation_mode"]
        self.top_p = self.dp_hparams["decoder"]["top_p"]
        self.temperature = self.dp_hparams["decoder"]["temperature"]
        assert self.generation_mode in ["greedy", "sample_top_p"], "Invalid generation mode.  Should be 'greedy' or 'sample_top_p'."
        logger.info("Duration predictor created.")

    @property
    def device(self):
        return self.encoder.device

    def encode_context(self, *, phonemes: torch.Tensor, ss: torch.Tensor, prompt_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(phonemes, ss=ss, prompt_embedding=prompt_embedding)

    def forward(self, *, duration: torch.Tensor, ss: torch.Tensor, pause: torch.Tensor, context: torch.Tensor):
        return self.decoder(duration=duration, pause=pause, ss=ss, context=context)

    def perform_training(self, *, duration: torch.Tensor, ss: torch.Tensor, pause: torch.Tensor, context: torch.Tensor) -> dict:
        assert duration.dim() == 2, "The input tensors should be 2D."
        assert duration.shape == pause.shape == ss.shape, "Shape of duration, pause, and ss should be the same."
        duration_input = prepare_for_autoregressive_training(sentences=duration, pad_id=self.decoder.pad_id, start_id=self.decoder.start_id, end_id=self.decoder.end_id, contexts=context, add_start_id_at_beginning=True, add_end_id_at_ending=True, device=self.device)
        pause_input = prepare_for_autoregressive_training(sentences=pause, pad_id=self.decoder.pause_pad_id, start_id=self.decoder.pause_start_id, end_id=self.decoder.pause_end_id, add_start_id_at_beginning=True, add_end_id_at_ending=True, device=self.device)
        ss_input = prepare_for_autoregressive_training(sentences=ss, pad_id=self.ss_tokenizer.pad_id, start_id=self.ss_tokenizer.start_id, end_id=self.ss_tokenizer.end_id, add_start_id_at_beginning=True, add_end_id_at_ending=True, device=self.device)
        decoded = self.forward(duration=duration_input["tokens"], pause=pause_input["tokens"], ss=ss_input["tokens"], context=duration_input["context"])
        duration_logits = decoded["duration_logits"]
        pause_logits = decoded["pause_logits"]
        ss_logits = decoded["ss_logits"]
        batch_size = duration_logits.shape[0]
        mini_batch_size = duration.shape[1] - 1
        next_dur_logits = torch.zeros((batch_size,), device=self.device)
        next_dur_labels = torch.zeros((batch_size,), device=self.device)
        next_pause_logits = torch.zeros((batch_size, self.decoder.pause_vocab_size), device=self.device)
        next_pause_labels = torch.zeros((batch_size,), dtype=torch.long, device=self.device)
        next_ss_logits = torch.zeros((batch_size, self.ss_tokenizer.vocab_size), device=self.device)
        next_ss_labels = torch.zeros((batch_size,), dtype=torch.long, device=self.device)
        for i in range(duration.shape[0]):
            for j in range(mini_batch_size):
                next_dur_logits[i * mini_batch_size + j] = duration_logits[i, j]
                next_dur_labels[i * mini_batch_size + j] = duration[i, j + 1]
                next_pause_logits[i * mini_batch_size + j] = pause_logits[i, j]
                next_pause_labels[i * mini_batch_size + j] = pause[i, j + 1]
                next_ss_logits[i * mini_batch_size + j] = ss_logits[i, j]
                next_ss_labels[i * mini_batch_size + j] = ss[i, j + 1]
        loss = F.mse_loss(next_dur_logits, next_dur_labels) + F.cross_entropy(next_pause_logits, next_pause_labels) + F.cross_entropy(next_ss_logits, next_ss_labels)
        return {
            "loss": loss,
            "duration_logits": duration_logits,
            "pause_logits": pause_logits,
            "ss_logits": ss_logits
        }

    @torch.inference_mode()
    def generate(self, *, context: torch.Tensor, leading: Optional[dict] = None, cache: Optional[InferAccelerationCache] = None, description: str = "Predicting durations for semantic...", mode: Optional[str] = None, max_gen_len: Optional[int] = None, top_p: Optional[float] = None, temperature: Optional[float] = None) -> dict:
        mode = mode if mode is not None else self.generation_mode
        assert mode in ["greedy", "sample_top_p"], "Invalid generation mode.  Should be 'greedy' or 'sample_top_p'."
        assert context.dim() == 3, "The context tensor should be 3D."
        assert context.shape[-1] == self.decoder.dim, ("The last dimension of context should be the same as the hidden dimension of the decoder.")
        if leading is None:
            leading = {
                "duration": torch.full((context.shape[0], 1), self.decoder.start_id, dtype=torch.long, device=self.device),
                "pause": torch.full((context.shape[0], 1), self.decoder.pause_start_id, dtype=torch.long, device=self.device),
                "ss": torch.full((context.shape[0], 1), self.ss_tokenizer.start_id, dtype=torch.long, device=self.device)
            }
            leading_len = 1
        else:
            for k in ["duration", "pause", "ss"]:
                assert k in leading, f"Leading should contain key '{k}'."
                assert leading[k].dim() == 2, "The leading tensors should be 2D."
                assert leading[k].shape[0] == context.shape[0], "The batch size of leading should be the same as context."
            assert leading["duration"].shape[1] == leading["pause"].shape[1] == leading["ss"].shape[1], "The sequence length of leading tokens should be the same."
            leading_len = leading["duration"].shape[1]
        if max_gen_len is None:
            max_gen_len = int(self.default_seq_len_ratio * context.shape[1])
        assert max_gen_len + leading_len <= self.max_seq_len, "The generated sequence length is too long."
        total_len = leading_len + max_gen_len
        duration_tokens = torch.full((context.shape[0], total_len), self.decoder.pad_id, dtype=torch.long, device=self.device)
        pause_tokens = torch.full((context.shape[0], total_len), 3, dtype=torch.long, device=self.device)
        ss_tokens = torch.full((context.shape[0], total_len), self.ss_tokenizer.pad_id, dtype=torch.long, device=self.device)
        duration_tokens[:, :leading_len] = leading["duration"]
        pause_tokens[:, :leading_len] = leading["pause"]
        ss_tokens[:, :leading_len] = leading["ss"]
        duration_mask = (duration_tokens != self.decoder.pad_id)
        pause_mask = (pause_tokens != self.decoder.pause_pad_id)
        ss_mask = (ss_tokens != self.ss_tokenizer.pad_id)
        if cache is None:
            cache = InferAccelerationCache(self.decoder.args, self.device)
        if top_p is None:
            top_p = self.top_p
        if temperature is None:
            temperature = self.temperature
        eos_reached = torch.full((context.shape[0],), False, dtype=torch.bool, device=self.device)
        prev_pos = 0
        for cur_pos in tqdm(range(leading_len, total_len), desc=description):
            decoded = self.decoder(duration=duration_tokens[:, prev_pos:cur_pos], pause=pause_tokens[:, prev_pos:cur_pos], ss=ss_tokens[:, prev_pos:cur_pos], context=context, cache=cache, start_pos=prev_pos)
            duration_logits = decoded["duration_logits"]
            pause_logits = decoded["pause_logits"]
            ss_logits = decoded["ss_logits"]
            duration_tokens[:, cur_pos] = torch.where(duration_mask[:, cur_pos], duration_tokens[:, cur_pos], duration_logits[:, -1].long())
            if mode == "greedy":
                pause_tokens[:, cur_pos] = torch.where(pause_mask[:, cur_pos], pause_tokens[:, cur_pos], torch.argmax(pause_logits[:, -1], dim=-1))
                ss_tokens[:, cur_pos] = torch.where(ss_mask[:, cur_pos], ss_tokens[:, cur_pos], torch.argmax(ss_logits[:, -1], dim=-1))
            else:
                probs = safe_softmax(pause_logits[:, -1] / temperature, dim=-1)
                pause_tokens[:, cur_pos] = torch.where(pause_mask[:, cur_pos], pause_tokens[:, cur_pos], sample_top_p(probs, p=top_p))
                probs = safe_softmax(ss_logits[:, -1] / temperature, dim=-1)
                ss_tokens[:, cur_pos] = torch.where(ss_mask[:, cur_pos], ss_tokens[:, cur_pos], sample_top_p(probs, p=top_p))
            eos_reached |= (~duration_mask[:, cur_pos] & ~pause_mask[:, cur_pos] & ~ss_mask[:, cur_pos] & (pause_tokens[:, cur_pos] == self.decoder.pause_end_id) & (ss_tokens[:, cur_pos] == self.ss_tokenizer.end_id))
            prev_pos = cur_pos
            if torch.all(eos_reached):
                break
        return {
            "duration": duration_tokens,
            "pause": pause_tokens,
            "ss": ss_tokens
        }

    @property
    def start_id(self):
        return self.decoder.start_id

    @property
    def end_id(self):
        return self.decoder.end_id

    @property
    def pad_id(self):
        return self.decoder.pad_id

    @property
    def sep_id(self):
        return self.decoder.sep_id

    @property
    def pause_vocab_size(self):
        return self.decoder.pause_vocab_size

    @property
    def pause_start_id(self):
        return self.decoder.pause_start_id

    @property
    def pause_end_id(self):
        return self.decoder.pause_end_id

    @property
    def pause_pad_id(self):
        return self.decoder.pause_pad_id

    @property
    def ss_vocab_size(self):
        return self.ss_tokenizer.vocab_size