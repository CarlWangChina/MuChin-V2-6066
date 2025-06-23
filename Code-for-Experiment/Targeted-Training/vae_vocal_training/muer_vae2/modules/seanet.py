import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from typing import List, Dict, Tuple, Any, Optional
from abc import ABC, abstractmethod
from ama_prof_divi_common.utils import get_logger
from .resnet import ResBlockSequence
from .lstm import SLSTM
from .utils import get_padding
from .swish import Swish
from Code_for_Experiment.Targeted_Training.vae_vocal_training.ama-prof-divi_vae2.modules.snake import SnakeBeta

logger = get_logger(__name__)

class SEANetBaseModule(nn.Module, ABC):
    def __init__(self,
                 num_channels: int,
                 latent_dim: int,
                 num_initial_filters: int,
                 resblock_kernel_sizes: List[int],
                 resblock_dilation_sizes: List[List[int]],
                 activation: str = "LeakyReLU",
                 activation_params: Optional[Dict[str, Any]] = None,
                 final_activation: str = "Tanh",
                 norm: str = "BatchNorm1d",
                 norm_params: Optional[Dict[str, Any]] = None,
                 padding_mode="reflect",
                 dropout: float = 0.0):
        super(SEANetBaseModule, self).__init__()
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        self.num_initial_filters = num_initial_filters
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.activation = activation
        self.activation_params = activation_params or {"negative_slope": 0.1}
        self.final_activation = final_activation
        self.norm = norm
        self.norm_params = norm_params or {}
        self.padding_mode = padding_mode
        self.dropout = dropout

    def get_activation(self) -> nn.Module:
        return getattr(nn, self.activation)(**self.activation_params)

    def get_final_activation(self) -> nn.Module:
        if self.final_activation == 'Snake':
            logger.info('Init Snake as activation')
            return SnakeBeta(self.latent_dim * 2)
        if self.final_activation == "Swish":
            print('Init Swish as activation')
            return Swish()
        return getattr(nn, self.final_activation)()

    def get_norm(self,
                 num_channels: int) -> nn.Module:
        if self.norm is None:
            return nn.Identity()
        return getattr(nn, self.norm)(num_channels, **self.norm_params)

    def get_res_blocks(self,
                       num_channels: int) -> nn.Module:
        return ResBlockSequence(num_channels=num_channels,
                                kernel_sizes=self.resblock_kernel_sizes,
                                dilations=self.resblock_dilation_sizes,
                                activation=self.activation,
                                activation_params=self.activation_params,
                                norm=self.norm,
                                norm_params=self.norm_params,
                                padding_mode=self.padding_mode)

    def get_dropout(self) -> nn.Module:
        class ConditionalDropout(nn.Module):
            def __init__(self, rate):
                super(ConditionalDropout, self).__init__()
                self.rate = rate

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if self.training and self.rate > 0.0:
                    return nn.functional.dropout(x, p=self.rate)
                return x
        return ConditionalDropout(self.dropout)

    @abstractmethod
    def remove_weight_norm_(self):
        ...

class SEANetEncoder(SEANetBaseModule):
    def __init__(self,
                 num_channels: int,
                 latent_dim: int,
                 num_initial_filters: int,
                 upsampling_ratios: List[int],
                 upsampling_kernel_sizes: List[int],
                 resblock_kernel_sizes: List[int],
                 resblock_dilation_sizes: List[List[int]],
                 initial_kernel_size: int = 7,
                 last_kernel_size: int = 7,
                 num_lstm_layers: int = 1,
                 activation: str = "LeakyReLU",
                 activation_params: Optional[Dict[str, Any]] = None,
                 final_activation: str = "Mish",
                 norm: str = "BatchNorm1D",
                 norm_params: Optional[Dict[str, Any]] = None,
                 padding_mode="reflect",
                 dropout: float = 0.0):
        super(SEANetEncoder, self).__init__(
            num_channels=num_channels,
            latent_dim=latent_dim,
            num_initial_filters=num_initial_filters,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            activation=activation,
            activation_params=activation_params,
            final_activation=final_activation,
            norm=norm,
            norm_params=norm_params,
            padding_mode=padding_mode,
            dropout=dropout)
        assert len(upsampling_ratios) == len(upsampling_kernel_sizes), \
            "The length of upsampling_ratios and upsampling_kernel_sizes must be the same."
        downsampling_ratios = upsampling_ratios[::-1]
        downsampling_kernel_sizes = upsampling_kernel_sizes[::-1]
        for i in range(len(downsampling_kernel_sizes)):
            downsampling_kernel_sizes[i] = downsampling_kernel_sizes[i] + 1 \
                if downsampling_kernel_sizes[i] % 2 == 0 else downsampling_kernel_sizes[i]
        self.input_layer = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels=num_channels,
                                  out_channels=num_initial_filters,
                                  kernel_size=initial_kernel_size,
                                  padding=get_padding(initial_kernel_size),
                                  padding_mode=self.padding_mode)),
            self.get_norm(num_initial_filters))
        module_list = nn.ModuleList()
        current_num_filters = num_initial_filters
        for i, ratio in enumerate(downsampling_ratios):
            module_list.append(nn.Sequential(
                self.get_res_blocks(current_num_filters),
                self.get_activation(),
                self.get_dropout(),
                weight_norm(nn.Conv1d(in_channels=current_num_filters,
                                      out_channels=current_num_filters * ratio,
                                      kernel_size=downsampling_kernel_sizes[i],
                                      stride=ratio,
                                      padding=get_padding(downsampling_kernel_sizes[i]),
                                      padding_mode=self.padding_mode)),
                self.get_norm(current_num_filters * ratio)
            ))
            current_num_filters *= ratio
        self.resnet_layers = nn.Sequential(*module_list)
        if num_lstm_layers > 0:
            self.lstm_layer = SLSTM(current_num_filters, num_layers=num_lstm_layers)
        else:
            self.lstm_layer = None
        self.final_layer = nn.Sequential(
            self.get_activation(),
            self.get_dropout(),
            weight_norm(nn.Conv1d(in_channels=current_num_filters,
                                  out_channels=latent_dim * 2,
                                  kernel_size=last_kernel_size,
                                  padding=get_padding(last_kernel_size),
                                  padding_mode=self.padding_mode)),
            self.get_norm(latent_dim * 2),
            self.get_final_activation()
        )
        self.output_layer_mean = nn.Conv1d(in_channels=latent_dim * 2,
                                           out_channels=latent_dim,
                                           kernel_size=3,
                                           padding=get_padding(3),
                                           padding_mode=self.padding_mode)
        self.output_layer_logvar = nn.Conv1d(in_channels=latent_dim * 2,
                                             out_channels=latent_dim,
                                             kernel_size=3,
                                             padding=get_padding(3),
                                             padding_mode=self.padding_mode)
        nn.init.kaiming_normal_(self.input_layer[0].weight)
        nn.init.zeros_(self.input_layer[0].bias)
        for layer in self.resnet_layers:
            nn.init.kaiming_normal_(layer[3].weight)
            nn.init.zeros_(layer[3].bias)
        nn.init.kaiming_normal_(self.final_layer[2].weight)
        nn.init.zeros_(self.final_layer[2].bias)
        nn.init.kaiming_normal_(self.output_layer_mean.weight)
        nn.init.zeros_(self.output_layer_mean.bias)
        nn.init.kaiming_normal_(self.output_layer_logvar.weight)
        nn.init.zeros_(self.output_layer_logvar.bias)

    def remove_weight_norm_(self):
        remove_parametrizations(self.input_layer[0], "weight")
        for layer in self.resnet_layers:
            layer[0].remove_weight_norm_()
            remove_parametrizations(layer[3], "weight")
        remove_parametrizations(self.final_layer[2], "weight")

    def forward(self,
                x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_layer(x)
        x = self.resnet_layers(x)
        if self.lstm_layer is not None:
            x = self.lstm_layer(x)
        x = self.final_layer(x)
        mean = self.output_layer_mean(x)
        logvar = self.output_layer_logvar(x)
        return mean, logvar

class SEANetDecoder(SEANetBaseModule):
    def __init__(self,
                 num_channels: int,
                 latent_dim: int,
                 num_initial_filters: int,
                 upsampling_ratios: List[int],
                 upsampling_kernel_sizes: List[int],
                 resblock_kernel_sizes: List[int],
                 resblock_dilation_sizes: List[List[int]],
                 initial_kernel_size: int = 7,
                 last_kernel_size: int = 7,
                 num_lstm_layers: int = 1,
                 activation: str = "LeakyReLU",
                 activation_params: Optional[Dict[str, Any]] = None,
                 final_activation: str = "Tanh",
                 norm: str = "BatchNorm1D",
                 norm_params: Optional[Dict[str, Any]] = None,
                 padding_mode="reflect",
                 dropout: float = 0.0):
        super(SEANetDecoder, self).__init__(
            num_channels=num_channels,
            latent_dim=latent_dim,
            num_initial_filters=num_initial_filters,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            activation=activation,
            activation_params=activation_params,
            final_activation=final_activation,
            norm=norm,
            norm_params=norm_params,
            padding_mode=padding_mode,
            dropout=dropout)
        assert len(upsampling_ratios) == len(upsampling_kernel_sizes), \
            "The length of upsampling_ratios and upsampling_kernel_sizes must be the same."
        current_num_filters = num_initial_filters
        for ratio in upsampling_ratios:
            current_num_filters *= ratio
        self.input_layer = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels=latent_dim,
                                  out_channels=current_num_filters,
                                  kernel_size=last_kernel_size,
                                  padding=get_padding(last_kernel_size),
                                  padding_mode=self.padding_mode)),
            self.get_norm(current_num_filters))
        if num_lstm_layers > 0:
            self.lstm_layer = SLSTM(current_num_filters, num_layers=num_lstm_layers)
        else:
            self.lstm_layer = None
        module_list = nn.ModuleList()
        for i, ratio in enumerate(upsampling_ratios):
            module_list.append(nn.Sequential(
                self.get_activation(),
                self.get_dropout(),
                weight_norm(nn.ConvTranspose1d(in_channels=current_num_filters,
                                               out_channels=current_num_filters // ratio,
                                               kernel_size=upsampling_kernel_sizes[i],
                                               stride=ratio,
                                               padding=(upsampling_kernel_sizes[i] - ratio) // 2)),
                self.get_norm(current_num_filters // ratio),
                self.get_res_blocks(current_num_filters // ratio)
            ))
            current_num_filters //= ratio
        self.resnet_layers = nn.Sequential(*module_list)
        self.final_layer = nn.Sequential(
            self.get_activation(),
            self.get_dropout(),
            weight_norm(nn.Conv1d(in_channels=current_num_filters,
                                  out_channels=num_channels,
                                  kernel_size=initial_kernel_size,
                                  padding=get_padding(initial_kernel_size),
                                  padding_mode=self.padding_mode)),
            self.get_final_activation()
        )
        nn.init.kaiming_normal_(self.input_layer[0].weight)
        nn.init.zeros_(self.input_layer[0].bias)
        for layer in self.resnet_layers:
            nn.init.kaiming_normal_(layer[2].weight)
            nn.init.zeros_(layer[2].bias)
        nn.init.kaiming_normal_(self.final_layer[2].weight)
        nn.init.zeros_(self.final_layer[2].bias)

    def remove_weight_norm_(self):
        remove_parametrizations(self.input_layer[0], "weight")
        for layer in self.resnet_layers:
            remove_parametrizations(layer[2], "weight")
            layer[4].remove_weight_norm_()
        remove_parametrizations(self.final_layer[2], "weight")

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        if self.lstm_layer is not None:
            x = self.lstm_layer(x)
        x = self.resnet_layers(x)
        x = self.final_layer(x)
        return x