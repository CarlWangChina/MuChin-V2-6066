import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from typing import Union, Callable, Optional, Tuple, List
from music_dit.utils import readonly, get_hparams
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.modules.vae.training.model import VAEModelForTraining

ADVERSARIAL_LOSSES = frozenset(['mse', 'hinge', 'hinge2'])
AdvLossType = Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]
FeatLossType = Union[nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]

class AdversarialLoss(nn.Module):
    def __init__(self,
                 model: Union[VAEModelForTraining, deepspeed.DeepSpeedEngine],
                 loss_real: AdvLossType,
                 loss_fake: AdvLossType,
                 loss_feat: Optional[FeatLossType] = None,
                 normalize: bool = True):
        super().__init__()
        self.model = model
        self.loss_real = loss_real
        self.loss_fake = loss_fake
        self.loss_feat = loss_feat
        self.normalize = normalize

    def forward(self,
                fake: torch.Tensor,
                real: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        adv = torch.tensor(0., device=fake.device)
        feat = torch.tensor(0., device=fake.device)
        fake_loss = torch.tensor(0., device=fake.device)
        real_loss = torch.tensor(0., device=fake.device)
        all_logits_fake, all_fmap_fake = self.model.discriminate(fake)
        all_logits_real, all_fmap_real = self.model.discriminate(real)
        n_sub_adversaries = len(all_logits_fake)
        for logit_fake, logit_real in zip(all_logits_fake, all_logits_real):
            _fake_loss = self.loss_fake(logit_fake)
            _real_loss = self.loss_real(logit_real)
            adv += _fake_loss + _real_loss
            fake_loss += _fake_loss
            real_loss += _real_loss
        if self.loss_feat:
            for fmap_fake, fmap_real in zip(all_fmap_fake, all_fmap_real):
                feat += self.loss_feat(fmap_fake, fmap_real)
        if self.normalize:
            adv /= n_sub_adversaries
            feat /= n_sub_adversaries
        return adv, feat, fake_loss, real_loss

def get_adv_criterion(loss_type: str) -> Callable:
    assert loss_type in ADVERSARIAL_LOSSES
    if loss_type == 'mse':
        return mse_loss
    elif loss_type == 'hinge':
        return hinge_loss
    elif loss_type == 'hinge2':
        return hinge2_loss
    raise ValueError('Unsupported loss')

def get_fake_criterion(loss_type: str) -> Callable:
    assert loss_type in ADVERSARIAL_LOSSES
    if loss_type == 'mse':
        return mse_fake_loss
    elif loss_type in ['hinge', 'hinge2']:
        return hinge_fake_loss
    raise ValueError('Unsupported loss')

def get_real_criterion(loss_type: str) -> Callable:
    assert loss_type in ADVERSARIAL_LOSSES
    if loss_type == 'mse':
        return mse_real_loss
    elif loss_type in ['hinge', 'hinge2']:
        return hinge_real_loss
    raise ValueError('Unsupported loss')

def mse_real_loss(x: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x, torch.tensor(1., device=x.device).expand_as(x))

def mse_fake_loss(x: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x, torch.tensor(0., device=x.device).expand_as(x))

def hinge_real_loss(x: torch.Tensor) -> torch.Tensor:
    return -torch.mean(torch.min(x - 1, torch.tensor(0., device=x.device).expand_as(x)))

def hinge_fake_loss(x: torch.Tensor) -> torch.Tensor:
    return -torch.mean(torch.min(-x - 1, torch.tensor(0., device=x.device).expand_as(x)))

def mse_loss(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return torch.tensor([0.0], device=x.device)
    return F.mse_loss(x, torch.tensor(1., device=x.device).expand_as(x))

def hinge_loss(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return torch.tensor([0.0], device=x.device)
    return -x.mean()

def hinge2_loss(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return torch.tensor([0.0])
    return -torch.mean(torch.min(x - 1, torch.tensor(0., device=x.device).expand_as(x)))

class FeatureMatchingLoss(nn.Module):
    def __init__(self, loss: nn.Module = torch.nn.L1Loss(), normalize: bool = True):
        super().__init__()
        self.loss = loss
        self.normalize = normalize

    def forward(self,
                fmap_fake: List[torch.Tensor],
                fmap_real: List[torch.Tensor]) -> torch.Tensor:
        assert len(fmap_fake) == len(fmap_real) and len(fmap_fake) > 0
        feat_loss = torch.tensor(0., device=fmap_fake[0].device)
        feat_scale = torch.tensor(0., device=fmap_fake[0].device)
        n_fmaps = 0
        for (feat_fake, feat_real) in zip(fmap_fake, fmap_real):
            assert feat_fake.shape == feat_real.shape
            n_fmaps += 1
            feat_loss += self.loss(feat_fake, feat_real)
            feat_scale += torch.mean(torch.abs(feat_real))
        if self.normalize:
            feat_loss /= n_fmaps
        return feat_loss

def get_adversarial_loss(model: Union[VAEModelForTraining, deepspeed.DeepSpeedEngine],
                         device: Optional[torch.device] = None) -> nn.Module:
    hparams = get_hparams()
    feat_loss_name = hparams.vae.training.adversarial.feat_loss
    assert feat_loss_name in ['l1', 'l2'], f'Unsupported feature matching loss: {feat_loss_name}'
    loss = torch.nn.L1Loss() if feat_loss_name == 'l1' else torch.nn.MSELoss()
    feat_loss = FeatureMatchingLoss(loss=loss, normalize=hparams.vae.training.adversarial.normalize)
    adv_loss = AdversarialLoss(model=model,
                               loss_real=get_real_criterion(hparams.vae.training.adversarial.adv_loss),
                               loss_fake=get_fake_criterion(hparams.vae.training.adversarial.adv_loss),
                               loss_feat=feat_loss,
                               normalize=hparams.vae.training.adversarial.normalize)
    if device is not None:
        adv_loss.to(device)
    return adv_loss