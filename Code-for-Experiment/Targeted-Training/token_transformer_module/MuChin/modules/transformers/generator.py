import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from tqdm import tqdm
from einops import rearrange
from ama_prof_divi.utils import sample_top_p, safe_softmax
from ama_prof_divi.utils.logging import get_logger
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.modules.transformers.model_args import TransformerModelArgs
from .transformers import TransformerDecoder
from .acceleration import InferAccelerationCache
import Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.modules.diffusion.losses.training as training

_losslogger = get_logger(__name__)

class Generator(nn.Module):
    def __init__(self, args: TransformerModelArgs, *, start_id: int, pad_id: int, end_id: int = -1, sep_id: int = -1, device: str or torch.device = "cpu"):
        super(Generator, self).__init__()
        self.args = args
        self.start_id = start_id
        self.sep_id = sep_id
        self.pad_id = pad_id
        self.end_id = end_id
        self.model = TransformerDecoder(args, device=device)

    @property
    def device(self):
        return self.model.device

    def forward(self, all_tokens: torch.Tensor, context: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, cache: Optional[InferAccelerationCache] = None, start_pos: int = 0, pos_bias: int or list or torch.Tensor = 0, pos_bias_k: int or list or torch.Tensor = 0) -> dict:
        assert all_tokens.dim() == 2 or all_tokens.dim() == 3, "The input tensor must be a 2D tensor (batches, seq_len), or a 3D tensor (batches, seq_len, num_q)."
        assert all_tokens.shape[1] <= self.args.max_seq_len, "The total length of tokens exceeds the maximum sequence length."
        assert all_tokens.shape[1] > 0, "The total length of tokens must be greater than 0."
        if all_tokens.dim() == 3:
            assert all_tokens.shape[2] == self.args.num_quantization_groups, f"The  all_tokens.shape[2] ({all_tokens.shape[2]}) must be the number of quantization groups."
        assert all_tokens.dtype == torch.long or all_tokens.dtype == torch.int, "The data type of the input tensor must be torch.int or torch.long."
        assert torch.max(all_tokens) < self.args.vocab_size, f"The maximum token ID ({torch.max(all_tokens)}) exceeds the vocabulary size ({self.args.vocab_size})."
        if context is not None:
            assert context.dim() == 3, "The context tensor must be a 3D tensor (batches, seq_len, dim)."
            assert context.shape[0] == all_tokens.shape[0], "The number of context tokens must be the same as the number of batches."
            assert context.shape[2] == self.args.dim, "The dimension of the context tensor must be the same as the model dimension."
        if labels is not None:
            assert labels.dim() == 2 or labels.dim() == 3, "The labels tensor must be a 2D tensor (batches, seq_len), or a 3D tensor (batches, seq_len, num_q)."
            if labels.dim() == 3:
                assert labels.shape == (all_tokens.shape[0], all_tokens.shape[1], self.args.num_quantization_groups), "The shape of the labels tensor must be (batch_size, seq_len, num_q)."
            else:
                assert labels.shape == (all_tokens.shape[0], all_tokens.shape[1]), "The shape of the labels tensor must be (batch_size, seq_len)."
            assert labels.dtype == torch.long or labels.dtype == torch.int, "The data type of the labels tensor must be torch.int or torch.long."
            assert torch.max(labels) < self.args.vocab_size, "The maximum token ID of the labels exceeds the vocabulary size."
        logits = self.model.forward(all_tokens, context=context, cache=cache, start_pos=start_pos, pos_bias=pos_bias, pos_bias_k=pos_bias_k)
        assert logits.dim() == 3 or logits.dim() == 4, ("The logits tensor must be a 3D tensor (batches, seq_len, vocab_size), or a 4D tensor (batches, seq_len, num_q, vocab_size.")
        if logits.dim() == 3:
            assert logits.shape == (all_tokens.shape[0], all_tokens.shape[1], self.model.out_dim), "The shape of the logits tensor must be (batch_size, seq_len, vocab_size)."
        else:
            assert logits.shape == (all_tokens.shape[0], all_tokens.shape[1], self.args.num_quantization_groups, self.model.out_dim), "The shape of the logits tensor must be (batch_size, seq_len, num_q, vocab_size)."
        if labels is not None:
            if logits.dim() == 4:
                logits_arranged = rearrange(logits, 'b s q d -> (b q) d s')
                labels_arranged = rearrange(labels, 'b s q -> (b q) s')
            else:
                logits_arranged = rearrange(logits, 'b s d -> b d s')
                labels_arranged = labels
            loss = F.cross_entropy(logits_arranged, labels_arranged, ignore_index=self.pad_id)
            return {"logits": logits, "loss": loss}
        else:
            return {"logits": logits}

    @torch.inference_mode()
    def generate(self, prompt_tokens: list or list[list] or torch.Tensor, context: Optional[torch.Tensor] = None, *, cache: InferAccelerationCache = None, description: str = "Autoregressive generating...", mode: str = "sample_top_p", max_gen_len: int = -1, temperature: float = 0.6, top_p: float = 0.9, pos_bias: int or list or torch.Tensor = 0, pos_bias_k: int or list or torch.Tensor = 0, output: str = "list") -> list or torch.Tensor:
        batches = len(prompt_tokens)
        if batches == 0:
            return [[]]
        self.eval()
        min_prompt_len = min([len(prompt) for prompt in prompt_tokens])
        max_prompt_len = max([len(prompt) for prompt in prompt_tokens])
        assert output in ["list", "tensor"], "Invalid output format.  Should be 'tensor' or 'list'."
        if min_prompt_len == 0:
            raise ValueError("Empty prompt is not allowed. At least, one token '<|ss_start|>' is required.")
        if context is not None:
            assert context.dim() == 3, "The context tensor must be a 3D tensor (batches, seq_len, dim)."
            assert context.shape[0] == batches, "The number of context tokens must be the same as the number of batches."
        assert max_prompt_len < self.args.max_seq_len, "The total length of prompt tokens exceeds the maximum sequence length."
        if max_gen_len < 0:
            max_gen_len = self.args.max_seq_len - max_prompt_len
        total_len = min(self.args.max_seq_len, max_prompt_len + max_gen_len)
        tokens = torch.full((batches, total_len, self.args.num_quantization_groups), self.pad_id, dtype=torch.long, device=self.device)
        if self.args.num_quantization_groups == 1:
            for k, t in enumerate(prompt_tokens):
                tokens[k, :len(t), 0] = torch.tensor(t, dtype=torch.long, device=self.device)
        else:
            for k, t in enumerate(prompt_tokens):
                tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=self.device)
        eos_reached = torch.full((batches, self.args.num_quantization_groups), False, dtype=torch.bool, device=self.device)
        input_text_mask = (tokens != self.pad_id)
        prev_pos = 0
        for cur_pos in tqdm(range(min_prompt_len, total_len), desc=description):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos, :], context=context, cache=cache, start_pos=prev_pos, pos_bias=pos_bias, pos_bias_k=pos_bias_k)
            if logits.dim() == 3:
                assert self.args.num_quantization_groups == 1
                logits = rearrange(logits, 'b s d -> b s 1 d')
            logits = logits[:, -1, :, :]
            assert logits.shape == (batches, self.args.num_quantization_groups, self.args.vocab_size)
            if mode == "sample_top_p" and temperature > 0:
                probs = safe_softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            elif mode == "greedy" or temperature <= 0:
                next_token = torch.argmax(logits, dim=-1)
            else:
                raise ValueError("Invalid generation mode '%s'." % mode)
            assert next_token.shape == (batches, self.args.num_quantization_groups)
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos] & (next_token == self.end_id) & (self.end_id >= 0))
            if cache is not None:
                prev_pos = cur_pos
            if torch.all(eos_reached):
                break
        out_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            start = len(prompt_tokens[i])
            toks = toks[start: start + max_gen_len]
            if self.end_id in toks:
                eos_idx = toks.index(self.end_id)
                toks = toks[:eos_idx]
                if self.args.num_quantization_groups > 1:
                    out_tokens.append(toks)
            else:
                out_tokens.append([t[0] for t in toks])
        if output == "list":
            return out_tokens
        else:
            num_batches = len(out_tokens)
            max_len = max([len(t) for t in out_tokens])
            out_tensor = torch.full((num_batches, max_len), self.pad_id, dtype=torch.long, device=self.device)
            for i, toks in enumerate(out_tokens):
                out_tensor[i, :len(toks)] = torch.tensor(toks, dtype=torch.long, device=self.device)
            return out_tensor

    @property
    def embedding(self):
        return self.model.embedding

    def prepare_for_autoregressive_training(self, *, sentences: list or list[torch.Tensor] or torch.Tensor, contexts: Optional[torch.Tensor or list[torch.Tensor]] = None, prompt_len: int = 0, start_id: Optional[int] = None, sep_id: Optional[int] = None, pad_id: Optional[int] = None, add_start_id_at_beginning: bool = False, insert_sep_id_after_prompt: bool = False, device: Optional[str or torch.device] = None) -> dict:
        return training.prepare_for_autoregressive_training(sentences=sentences, pad_id=pad_id if pad_id is not None else self.pad_id, start_id=start_id if start_id is not None else self.start_id, sep_id=sep_id if sep_id is not None else self.sep_id, contexts=contexts, prompt_len=prompt_len, add_start_id_at_beginning=add_start_id_at_beginning, insert_sep_id_after_prompt=insert_sep_id_after_prompt, device=device if device is not None else self.device)