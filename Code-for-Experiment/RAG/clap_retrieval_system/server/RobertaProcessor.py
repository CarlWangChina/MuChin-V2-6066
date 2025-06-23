import librosa
import torch
from transformers import BertTokenizer, BertModel

MODEL_PATH = "hfl/chinese-roberta-wwm-ext-large"

class RobertaProcessor:
    def __init__(self, device="cuda") -> None:
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        self.model = BertModel.from_pretrained(MODEL_PATH).to(device)

    @torch.no_grad()
    def processText(self, text_data: list[str]):
        encoded_input = self.tokenizer(text_data, return_tensors="pt", padding=True).to(self.device)
        last_hidden_state = self.model(**encoded_input).last_hidden_state
        vec = torch.mean(last_hidden_state, dim=1)
        return vec

if __name__ == "__main__":
    p = RobertaProcessor()
    text_data = ["I love the contrastive learning", "I love the pretrain model"]
    print(p.processText(text_data).shape)