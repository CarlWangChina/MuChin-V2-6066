import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from typing import Tuple, List
from pathlib import Path
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.hparams import get_hparams
from .losses import discriminator_loss, generator_loss, feature_loss

LRELU_SLOPE = 0.1

def get_padding(kernel_size: int, dilation: int = 1):
    return int((kernel_size * dilation - dilation) / 2)

class DiscriminatorP(nn.Module):
    def __init__(self,
                 channels: int,
                 period: int,
                 kernel_size: int = 5,
                 stride: int = 3,
                 use_spectral_norm: bool = False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(channels, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight)
            nn.init.zeros_(conv.bias)
        nn.init.kaiming_normal_(self.conv_post.weight)
        nn.init.zeros_(self.conv_post.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for c in self.convs:
            x = c(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, channels: int):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(channels, 2),
            DiscriminatorP(channels, 3),
            DiscriminatorP(channels, 5),
            DiscriminatorP(channels, 7),
            DiscriminatorP(channels, 11)
        ])

    def forward(self,
                y: torch.Tensor,
                y_hat: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class DiscriminatorS(nn.Module):
    def __init__(self,
                 channels: int,
                 use_spectral_norm: bool = False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(channels, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight)
            nn.init.zeros_(conv.bias)
        nn.init.kaiming_normal_(self.conv_post.weight)
        nn.init.zeros_(self.conv_post.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, channels: int):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(channels, use_spectral_norm=True),
            DiscriminatorS(channels),
            DiscriminatorS(channels),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self,
                y: torch.Tensor,
                y_hat: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        root_dir = Path(__file__).parent.parent.parent.parent
        hparams = get_hparams(root_dir)
        num_channels = hparams.vae.num_channels
        self.mpd = MultiPeriodDiscriminator(num_channels)
        self.msd = MultiScaleDiscriminator(num_channels)

    def forward(self,
                y: torch.Tensor,
                y_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_df_rs, y_df_gs, fmap_f_rs, fmap_f_gs = self.mpd(y, y_hat)
        y_ds_rs, y_ds_gs, fmap_s_rs, fmap_s_gs = self.msd(y, y_hat)
        loss_disc_f, _, _ = discriminator_loss(y_df_rs, y_df_gs)
        loss_disc_s, _, _ = discriminator_loss(y_ds_rs, y_ds_gs)
        loss_disc_all = loss_disc_f + loss_disc_s
        loss_gen_f, _ = generator_loss(y_df_gs)
        loss_gen_s, _ = generator_loss(y_ds_gs)
        loss_gen_all = loss_gen_f + loss_gen_s
        loss_fm_f = feature_loss(fmap_f_rs, fmap_f_gs)
        loss_fm_s = feature_loss(fmap_s_rs, fmap_s_gs)
        loss_fm_all = loss_fm_f + loss_fm_s
        return loss_disc_all, loss_gen_all, loss_fm_all