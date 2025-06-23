import torch
import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp.rnn
import nn

class SSTokenizer(nn.Module):
    def __init__(self, hparams: dict):
        super(SSTokenizer, self).__init__()
        self.hparams = hparams
        st = hparams["ama-prof-divi"]["models"]["lyrics"]["tokenizer"]["special_tokens"]
        special_tokens = []
        for tok in st:
            if tok.startswith("<|ss_"):
                special_tokens.append(tok)
        st = sorted(special_tokens)
        self.special_tokens = {}
        for i, tok in enumerate(st):
            self.special_tokens[tok] = i + 1
        self.special_tokens_inv = {}
        for k, v in self.special_tokens.items():
            self.special_tokens_inv[v] = k

    @property
    def vocab_size(self):
        return len(self.special_tokens) + 1

    def encode(self, ss: [str] or str, *, pad_id: int = 0) -> torch.Tensor:
        if isinstance(ss, str):
            ss = [ss]
        num_batches = len(ss)
        tokens = []
        max_seq_len = 0
        for ss_str in ss:
            ss_split = ss_str.split()
            tok = []
            for t in ss_split:
                t = t.strip().lower()
                if t in self.special_tokens:
                    tok.append(self.special_tokens[t])
                else:
                    assert t == "", f"Unknown song structure token: '{t}'"
            if max_seq_len < len(tok):
                max_seq_len = len(tok)
            tokens.append(tok)
        result = torch.full((num_batches, max_seq_len), pad_id, dtype=torch.long)
        for i, tok in enumerate(tokens):
            result[i, :len(tok)] = torch.tensor(tok, dtype=torch.long)
        return result

    def forward(self, ss: [str] or str, *, pad_id: int = 0) -> torch.Tensor:
        return self.encode(ss, pad_id=pad_id)

    def decode(self, token_ids: torch.Tensor, *, pad_id: int = 0) -> [str]:
        assert token_ids.dim() == 2, "The input tensor should be 2D."
        assert token_ids.dtype == torch.long, "The input tensor should be of type torch.long."
        assert torch.min(token_ids) >= 0, "The input tensor should not contain negative values."
        assert torch.max(token_ids) < self.vocab_size, ("The input tensor should not contain values larger than "
                                                        "the vocab size.")
        result = []
        for i in range(token_ids.shape[0]):
            tok = []
            for j in range(token_ids.shape[1]):
                if token_ids[i, j] == pad_id:
                    continue
                tok.append(self.special_tokens_inv[token_ids[i, j]])
            result.append(" ".join(tok))
        return result

    @property
    def start_id(self):
        return self.encode('<|ss_start|>')[0, 0]

    @property
    def end_id(self):
        return self.encode('<|ss_end|>')[0, 0]

    @property
    def pad_id(self):
        return self.encode('<|ss_pad|>')[0, 0]

    @property
    def sep_id(self):
        return self.encode('<|ss_sep|>')[0, 0]

    @property
    def unk_id(self):
        return self.encode('<|ss_unk|>')[0, 0]