import torch
import torch.nn as nn
from music_dit.utils import get_logger, get_hparams
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.modules.vae.vae import VAEModel
from .adversarial import MultiScaleSTFTDiscriminator, MultiDiscriminatorOutputType

logger = get_logger(__name__)

class VAEModelForTraining(nn.Module):
    def __init__(self):
        super(VAEModelForTraining, self).__init__()
        hparams = get_hparams()
        device = hparams.device
        vae_model = VAEModel()
        self.encoder = vae_model.encoder
        self.decoder = vae_model.decoder
        self.chunk_length = vae_model.chunk_length
        self.num_channels = vae_model.num_channels
        self.discriminator = MultiScaleSTFTDiscriminator(
            in_channels=hparams.vae.num_channels,
            out_channels=hparams.vae.num_channels,
            filters=hparams.vae.training.msstftd.filters,
            n_ffts=hparams.vae.training.msstftd.n_ffts,
            hop_lengths=hparams.vae.training.msstftd.hop_lengths,
            win_lengths=hparams.vae.training.msstftd.win_lengths,
            norm=hparams.vae.training.msstftd.norm,
            activation_type=hparams.vae.training.msstftd.activation,
            activation_params=hparams.vae.training.msstftd.activation_params
        )
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_decode(x)

    def discriminate(self, x: torch.Tensor) -> MultiDiscriminatorOutputType:
        return self.discriminator(x)