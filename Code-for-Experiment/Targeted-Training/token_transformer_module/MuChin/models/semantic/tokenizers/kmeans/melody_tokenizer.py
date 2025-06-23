from pathlib import Path
from ama_prof_divi.utils import logging
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.models.lyrics.builder import get_lyrics_tokenizer
from .tokenizer_base import SemanticTokenizerBase

logger = logging.getLogger(__name__)

class MelodyTokenizer(SemanticTokenizerBase):
    """Melody tokenizer.    Args:        hparams (dict):         System-wide hyperparameters.        load_pretrained (bool): Whether to load the pretrained model.    """

    def __init__(self, hparams: dict, load_pretrained: bool = True):
        self.km_hparams = hparams["ama-prof-divi"]["models"]["semantic"]["tokenizer"]["kmeans"]
        super(MelodyTokenizer, self).__init__(n_dim=self.km_hparams["n_dim"],
                                              num_clusters=self.km_hparams["num_clusters"],
                                              cluster_batch_size=self.km_hparams["cluster_batch_size"],
                                              hparams=hparams,
                                              n_init=self.km_hparams["n_init"],
                                              random_state=self.km_hparams["random_state"],
                                              max_iter=self.km_hparams["max_iter"],
                                              lyrics_tokenizer=get_lyrics_tokenizer(hparams=hparams))
        if load_pretrained:
            checkpoints_dir = Path(self.hparams["ama-prof-divi"]["root_path"]).joinpath("checkpoints", "semantic")
            self.pretrained_joblib_file = checkpoints_dir.joinpath(self.km_hparams["pretrained_joblib"])
            self.try_load_pretrained_joblib(str(self.pretrained_joblib_file),
                                            self.km_hparams["pretrained_joblib_url"],
                                            self.km_hparams["pretrained_joblib_sha256"])
        if not self.is_fitted:
            logger.warning("The melody tokenizer model is not fitted. Please run `fit()` first.")