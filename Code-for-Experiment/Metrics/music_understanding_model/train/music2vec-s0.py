import sys
sys.path.append("./")
import torch
from models.mlp import MLP, F1_score
from Code_for_Experiment.Metrics.music_understanding_model.dataset import VecDatasetMusic2Vec
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
from omegaconf import OmegaConf

cfg = OmegaConf.load('configs/train.yaml')
num_epochs = cfg.model.music2vec_s0.num_epochs
save_path = cfg.model.music2vec_s0.save_path
device = cfg.model.music2vec_s0.device
dataset_index = cfg.model.music2vec_s0.dataset_index
loss_path = cfg.model.music2vec_s0.loss_path
output_len = cfg.model.music2vec_s0.output_len

ds = VecDatasetMusic2Vec(path=dataset_index, output_len=output_len, replace=cfg.model.music2vec_s0.dataset)
data_loader = torch.utils.data.DataLoader(ds, batch_size=cfg.model.music2vec_s0.batch_size, shuffle=True)

model = MLP(768, 8192, output_len).to(device)
criterion = BCEWithLogitsLoss(pos_weight=ds.weight).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
loss_file = open(loss_path, "w")

for epoch in range(num_epochs):
    with tqdm(total=len(data_loader), ncols=160) as pbar:
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            pbar.update()
            acc, recall, f1 = F1_score((outputs > 0.51).int(), targets)
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs} Loss: {loss.item()} acc:{acc} recall:{recall} f1:{f1}")
            loss_file.write(f"Epoch {epoch+1}/{num_epochs} Loss: {loss.item()} acc:{acc} recall:{recall} f1:{f1}\n")
            if epoch % 20 == 0:
                torch.save(model.state_dict(), f"{save_path}/{epoch}.ckpt")
loss_file.close()