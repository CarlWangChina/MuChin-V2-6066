import torch
import torch.nn as nn
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.hparams import get_hparams
from ama_prof_divi.models.lyrics import get_lyrics_to_phoneme_translator
from ama_prof_divi.modules.transformers import TransformerModelArgs, TransformerEncoder

class SongStructureEncoder(nn.Module):
    """ The song structure encoder.
    Args:
        hparams (dict): Hyper-parameters.
    """

    def __init__(self, hparams: dict = None):
        super(SongStructureEncoder, self).__init__()
        if hparams is None:
            hparams = get_hparams()
        self.hparams = hparams
        self.encoder_hparams = self.hparams["ama-prof-divi"]["models"]["lyrics"]["song_structure_encoder"]
        self.device = self.hparams["ama-prof-divi"]["device"]
        self.dim = self.encoder_hparams["dim"]
        self.lyrics2phoneme = get_lyrics_to_phoneme_translator(self.hparams)
        args = TransformerModelArgs(
            dim=self.dim,
            num_layers=self.encoder_hparams["num_layers"],
            num_heads=self.encoder_hparams["num_heads"],
            dropout=self.encoder_hparams["dropout"],
            max_seq_len=self.encoder_hparams["max_seq_len"],
            hidden_dim=self.encoder_hparams["hidden_dim"],
            vocab_size=self.lyrics2phoneme.phoneme_tokenizer.vocab_size
        )
        self.args = args
        self.encoder = TransformerEncoder(args, device=self.device)

    def forward(self, lyrics: [str]) -> torch.Tensor:
        """ Forward pass.
        Args:
            lyrics (list): List of lyrics, grouped by batches.
        Returns:
            torch.Tensor: Encoded lyrics. Shape: (batch_size, seq_len, self.dim).
        """
        phoneme_tokens = self.lyrics2phoneme(lyrics)
        return self.encoder(phoneme_tokens)