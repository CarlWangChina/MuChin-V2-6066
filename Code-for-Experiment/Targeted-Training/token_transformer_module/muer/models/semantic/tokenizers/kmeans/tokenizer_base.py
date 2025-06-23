import torch
import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp.rnn_compat
import joblib
import os
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans as KMeans
import logging
import download_file
from ama_prof_divi.models.lyrics import LyricsTokenizer
logger = logging.getLogger(__name__)
class SemanticTokenizerBase(torch.nn.Module):
    def __init__(self,                 n_dim: int,                 num_clusters: int,                 cluster_batch_size: int,                 *,                 hparams: dict,                 n_init: int or str = "auto",                 random_state: int or None = 0,                 max_iter: int = 1000,                 lyrics_tokenizer: LyricsTokenizer,                 **args):
        super(SemanticTokenizerBase, self).__init__()
        self.n_dim = n_dim
        self.num_clusters = num_clusters
        self.hparams = hparams
        self.kmeans = KMeans(n_clusters=num_clusters,                             n_init=n_init,                             batch_size=cluster_batch_size,                             random_state=random_state,                             max_iter=max_iter)
        self.special_tokens_dict = {}

        self.vocab_size = num_clusters
        self.num_quantizers = 1
        for token in sorted(lyrics_tokenizer.special_tokens_set()):
            self.special_tokens_dict[token] = self.vocab_size
            self.vocab_size += 1
        self.start_token = lyrics_tokenizer.start_token
        self.pad_token = lyrics_tokenizer.pad_token
        self.end_token = lyrics_tokenizer.end_token
        self.mask_token = lyrics_tokenizer.mask_token
        self.unknown_token = lyrics_tokenizer.unknown_token
    @property
    def is_fitted(self) -> bool:
        return hasattr(self, "kmeans") and hasattr(self.kmeans, "cluster_centers_")
    def fit(self, x: torch.Tensor):
        if self.is_fitted:
            raise RuntimeError("The model has been fitted already.")
        if x.dim() != 2:
            raise ValueError("The input data must be a 2D tensor.")
        if x.shape[1] != self.n_dim:
            raise ValueError(f"The input data must have {self.n_dim} columns.")
        logger.info(f"Fitting the model with {x.shape[0]} samples...")
        self.kmeans.partial_fit(x)
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("The model has not been fitted yet.")
        if x.dim() != 2:
            raise ValueError("The input data must be a 2D tensor.")
        if x.shape[1] != self.n_dim:
            raise ValueError(f"The input data must have {self.n_dim} columns.")
        return torch.Tensor(self.kmeans.predict(x))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)
    def save(self, path: str):
        if not self.is_fitted:
            raise RuntimeError("The model has not been fitted yet.")
        logger.info(f"Saving the model to {path}...")
        joblib.dump(self.kmeans, path)
    def load(self, path: str):
        logger.info(f"Loading the model from {path}...")
        self.kmeans = joblib.load(path)
    def is_special_token(self, token: int) -> bool:
        return self.num_clusters <= token < self.vocab_size
    def encode_special_token(self, token: str) -> int:
        if token not in self.special_tokens_dict:
            raise ValueError(f"{token} is not a special token.")
        return self.special_tokens_dict[token]
    @property
    def pad_id(self) -> int:
        return self.encode_special_token(self.pad_token)
    @property
    def start_id(self) -> int:
        return self.encode_special_token(self.start_token)
    @property
    def end_id(self) -> int:
        return self.encode_special_token(self.end_token)
    @property
    def mask_id(self) -> int:
        return self.encode_special_token(self.mask_token)
    @property
    def unknown_id(self) -> int:
        return self.encode_special_token(self.unknown_token)
    def try_load_pretrained_joblib(self,                                   pretrained_joblib_file: str,                                   pretrained_joblib_url: str,                                   pretrained_joblib_sha256: str) -> bool:
        joblib_dir = Path(pretrained_joblib_file).parent
        if joblib_dir.is_file():
            raise RuntimeError(f"{joblib_dir} exists and is not a directory.")
        if not joblib_dir.exists():
            os.makedirs(joblib_dir)
        if not Path(pretrained_joblib_file).exists():
            if pretrained_joblib_url is None or pretrained_joblib_url == "":
                logger.warning(f"Pretrained joblib file {Path(pretrained_joblib_file).name} does not exist.")
                return False
            logger.info(f"Downloading pretrained joblib file {Path(pretrained_joblib_file).name}...")
            download_file(url=pretrained_joblib_url,                          download_target=str(pretrained_joblib_file),                          expected_sha256=pretrained_joblib_sha256)
        self.load(pretrained_joblib_file)
        assert self.is_fitted
        return True