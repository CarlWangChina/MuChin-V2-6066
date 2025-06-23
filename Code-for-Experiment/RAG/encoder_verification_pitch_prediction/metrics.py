import torch
from utils import *

check = False

def accuracy(data1: torch.IntTensor, data2: torch.IntTensor):
    assert data1.shape == data2.shape
    data1 = data1.view(-1)
    data2 = data2.view(-1)
    acc = torch.sum(data1 == data2) / len(data1)
    return acc

def F1_score(data1: torch.IntTensor, data2: torch.IntTensor):
    assert data1.shape == data2.shape
    data1 = data1.view(-1)
    data2 = data2.view(-1)
    if check:
        print("data1")
        print(data1.tolist())
        print("data2")
        print(data2.tolist())
    acc = torch.sum(data1 == data2) / len(data1)
    if acc == 1:
        return torch.tensor(1.0)
    precision = (
        torch.sum(data1[data1 > 0] == data2[data1 > 0]) /
        (torch.sum(data1 > 0) + 1e-5)
    )
    recall = (
        torch.sum(data1[data2 > 0] == data2[data2 > 0]) /
        (torch.sum(data2 > 0) + 1e-5)
    )
    score = 2 * precision * recall / (precision + recall + 1e-5)
    return score

def eval_acc(model, dataloader, thresh=None, method="accuracy"):
    model.eval()
    total_acc = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for data in dataloader:
            tokens, melody = data
            tokens = tokens.to(device)
            melody = melody.to(device)
            out = model(tokens)
            pred = predict(out, thresh)
            if method == "accuracy":
                acc = accuracy(pred, melody)
            elif method == "F1-score":
                acc = F1_score(pred, melody)
            else:
                raise ValueError
            total_acc += acc
        total_acc /= len(dataloader)
    return total_acc

if __name__ == "__main__":
    import yaml
    from Code_for_Experiment.RAG.encoder_verification_pitch_prediction.dataset import get_data
    with open("config/Code-for-Experiment/Targeted-Training/hifigan_vocoder/configs/Code-for-Experiment/Targeted-Training/hifigan_vocoder/configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        print(config)
    model_config = config['model']
    data_config = config['data']
    train_config = config['train']
    model = torch.load("./model_save/test_model_0.pth", map_location="cuda:2")
    train_loader, valid_loader, test_loader = get_data(data_config)
    acc = eval_acc(model, valid_loader, 0.8, "F1-score")
    print(acc)