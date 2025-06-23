import torch
from torch import nn
import logging
from Code_for_Experiment.RAG.encoder_verification_pitch_prediction.cluster_embedding import ClusterEmbedding

logger = logging.getLogger(__name__)

class BaseEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        raise NotImplementedError

class MertEmbedding(BaseEmbedding):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class TrmEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        nhead = config['nhead']
        hidden_size = config['hidden_size']
        dropout = config['dropout']
        num_layers = config['num_layers']

        self.pe = PositionalEncoding(d_model, dropout)
        trm_encoder_layers = nn.TransformerEncoderLayer(d_model=d_model,
                                                        nhead=nhead,
                                                        dim_feedforward=hidden_size,
                                                        dropout=dropout,
                                                        batch_first=True)
        self.trm_encoder = nn.TransformerEncoder(trm_encoder_layers, num_layers)

    def forward(self, x):
        x = self.pe(x)
        out = self.trm_encoder(x)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=50000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class DenseDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_size = config['input_size']
        hidden_size = config['hidden_size']
        output_size = config['output_size']
        num_layers = config['num_layers']
        dropout = config['dropout']

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        num_layers -= 1
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size, output_size))
        self.dense_layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dense_layers(x)
        return out

class MelodyTranscriptionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        vocab_size = config['vocab_size']
        embed_size = config['embed_size']
        encoder_config = config['encoder']
        decoder_config = config['decoder']
        embedding = config["embedding"]
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        logger.debug(embedding["model"])

        if embedding["model"] == "cluster":
            self.embed = ClusterEmbedding(embedding["path"])
        elif embedding["model"] == "mert" or embedding["model"] == "MERT":
            self.embed = MertEmbedding()
        else:
            self.embed = nn.Embedding(vocab_size, embed_size)
        self.encoder = TrmEncoder(encoder_config)
        self.decoder = DenseDecoder(decoder_config)

    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        out = self.decoder(x)
        return out

class Tokens2PitchOnsetModel(MelodyTranscriptionModel):
    def __init__(self, config):
        super().__init__(config)
        self.onset_predictor = nn.Linear(self.embed_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        out1 = self.decoder(x)
        out2 = self.onset_predictor(x)
        out2 = self.sigmoid(out2)
        return out1, out2

if __name__ == '__main__':
    import yaml
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="path of the yaml format config file"
    )
    parser.set_defaults(
        config="./config/config.yaml"
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        print(config)
    data_config = config["data"]
    model_config = config["model"]
    my_model = Tokens2PitchOnsetModel(model_config)
    src = torch.randint(0, 100, (4, 24))
    print(src)
    out1, out2 = my_model(src)
    print(out1)
    print(src.shape, out1.shape)