import joblib
import sklearn
from torch import nn
import torch

class ClusterEmbedding(nn.Module):
    def __init__(self, path="./melody_cluster.joblib"):
        super().__init__()
        cluster = joblib.load(path)
        self.cluster_centers = torch.tensor(cluster.cluster_centers_)

    def forward(self, x):
        self.cluster_centers = self.cluster_centers.to(x.device)
        return self.cluster_centers[x]

class TestEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = ClusterEmbedding()

    def forward(self, x):
        return self.embed(x)

if __name__ == "__main__":
    path = "./melody_cluster.joblib"
    embed = joblib.load(path)
    print(type(embed))