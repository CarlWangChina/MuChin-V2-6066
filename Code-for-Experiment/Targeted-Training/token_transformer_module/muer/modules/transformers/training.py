import torch
from typing import Optional
from einops import rearrange
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.misc import merge_tensors

def prepare_for_autoregressive_training(*, sentences: [[int]] or [torch.Tensor] or torch.Tensor, pad_id: int = 0, start_id: Optional[int] = None, sep_id: Optional[int] = None, end_id: Optional[int] = None, contexts: Optional[torch.Tensor or [torch.Tensor]] = None, prompt_len: int = 0, add_start_id_at_beginning: bool = False, add_end_id_at_ending: bool = False, insert_sep_id_after_prompt: bool = False, device: str or torch.device = "cpu") -> dict:
    """Prepare for autoregressive training.
    Args:
        sentences ([[int]]):
            List of tokenized sentences. The total dimension should be 2 or 3. If the
            dimension is 2, the shape should be (batch_size, seq_len). If the dimension
            is 3, the shape should be (batch_size, seq_len, num_quantization_groups).
            The sentence tensor should start with start_id, and contain the prompt
            tokens, followed by the target tokens. Prompt tokens are optional. The
            end_id at the end of sentence is optional.
            For example,
            <start> prompt, prompt, <sep>, target, target, <end>
        contexts (torch.Tensor):
            Context vectors (optional). shape: (batch_size, context_len, dim)
        pad_id (int):                   ID of padding token. Defaults to 0.
        start_id (int):                 ID of start token. Used only when add_start_id_at_beginning is True.
        sep_id (int):                   ID of separator token. Used only when insert_sep_id_after_prompt is True.
        end_id (int):                   ID of end token. Used only when add_end_id_at_ending is True.
        prompt_len (int):               Length of prompt tokens. Since prompt is included in the input sentence,
            the length of sentence is the length of prompt tokens plus the length of
            target tokens.
        add_start_id_at_beginning:      Whether to add start ID at beginning of each sentence. Default to False.
        add_end_id_at_ending:           Whether to add end ID at ending of each sentence. Default to False.
        insert_sep_id_after_prompt:     Whether to insert separator ID after prompt tokens. Default to False.
            If no prompt tokens (prompt_len = 0), this option is ignored.
        device (str or torch.device):   Device to use (optional). Default to cpu.
    Returns:
        Dict:                           Dictionary of outputs.
        - tokens (torch.Tensor):        Tokens tensor. shape: (batch_size, seq_len), or (batch_size, seq_len,
            num_quantization_groups).
        - context (torch.Tensor):       Context vectors. shape: (batch_size, context_len, dim). Optional.
        - labels (torch.Tensor):        Labels tensor. shape: (batch_size, seq_len), or (batch_size, seq_len,
            num_quantization_groups).
    """

    num_batches = len(sentences)
    if isinstance(sentences, list):
        sentences = [torch.tensor(sent, dtype=torch.long, device=device) for sent in sentences]
    else:
        assert torch.is_tensor(sentences), "The input sentences must be a list, or a tensor."
    if contexts is not None:
        assert len(contexts) == num_batches, (f"The length of context ({len(contexts)}) must be the same of batch numbers ({num_batches}).")
    all_tokens_list = []
    all_labels_list = []
    all_context_list = []
    assert prompt_len >= 0
    if prompt_len == 0:
        insert_sep_id_after_prompt = False
    if add_start_id_at_beginning:
        assert start_id is not None and start_id >= 0, "The start_id must be greater than or equal to 0."
        prompt_len += 1
    else:
        assert prompt_len > 0, ("The length of prompt tokens must be greater than 0, if we do not add the start token at beginning. At least, the prompt should contains one token <start>.")
    if insert_sep_id_after_prompt:
        assert sep_id is not None and sep_id >= 0, "The sep_id must be greater than or equal to 0."
        prompt_len += 1
    if add_end_id_at_ending:
        assert end_id is not None and end_id >= 0, "The end_id must be greater than or equal to 0."
        prompt_len += 1
    for batch in range(num_batches):
        sentence = sentences[batch]
        if add_start_id_at_beginning:
            start = torch.full(sentence.shape[1:], start_id, dtype=torch.long, device=device).unsqueeze(0)
            sentence = torch.cat([start, sentence], dim=0)
        if insert_sep_id_after_prompt:
            sep = torch.full(sentence.shape[1:], sep_id, dtype=torch.long, device=device).unsqueeze(0)
            sentence = torch.cat([sentence[:prompt_len - 1], sep, sentence[prompt_len - 1:]], dim=0)
        if add_end_id_at_ending:
            end = torch.full(sentence.shape[1:], end_id, dtype=torch.long, device=device).unsqueeze(0)
            sentence = torch.cat([sentence, end], dim=0)
        seq_len = len(sentence)
        assert seq_len > prompt_len, (f"The length of sentence ({len(sentence)}) must be greater than the length of prompt tokens ({prompt_len}).")
        mini_batch_size = seq_len - prompt_len + 1
        if sentence.dim() == 1:
            all_tokens = torch.tensor(sentence, dtype=torch.long).expand(mini_batch_size, -1)
        else:
            assert sentence.dim() == 2, (f"The input sentence must be a 1D or 2D tensor. Actual dim is {sentence.dim()}.")
            all_tokens = torch.tensor(sentence, dtype=torch.long).expand(mini_batch_size, -1, -1)
            all_tokens = rearrange(all_tokens, 'b l q -> q b l')
        all_tokens = (torch.tril(all_tokens, diagonal=prompt_len - 1) + torch.triu(torch.ones_like(all_tokens) * pad_id, diagonal=prompt_len))
        if sentence.dim() == 2:
            all_tokens = rearrange(all_tokens, 'q b l -> b l q')
        labels = torch.full(all_tokens.shape, pad_id, dtype=torch.long, device=device)
        for i in range(mini_batch_size - 1):
            labels[i, i + prompt_len - 1] = all_tokens[i + 1, i + prompt_len]
        all_tokens = all_tokens[:mini_batch_size - 1, ...]
        labels = labels[:mini_batch_size - 1, ...]
        mini_batch_size -= 1
        all_tokens_list.append(all_tokens)
        all_labels_list.append(labels)
        if contexts is not None:
            ctx = contexts[batch].unsqueeze(0).expand(mini_batch_size, -1, -1)
            all_context_list.append(ctx)
    all_tokens = merge_tensors(all_tokens_list, padding=pad_id, dtype=torch.long, device=device)
    labels = merge_tensors(all_labels_list, padding=pad_id, dtype=torch.long, device=device)
    if contexts is not None:
        contexts = merge_tensors(all_context_list, padding=0.0, dtype=torch.float, device=device)
        return {
            "tokens": all_tokens,
            "context": contexts,
            "labels": labels
        }
    else:
        return {
            "tokens": all_tokens,
            "labels": labels
        }