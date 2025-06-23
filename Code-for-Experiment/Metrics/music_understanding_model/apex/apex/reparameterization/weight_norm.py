import torch
from torch.nn.parameter import Parameter
from ..fp16_utils import Fused_Weight_Norm
import time
from Code_for_Experiment.Metrics.music_understanding_model.apex.apex.reparameterization.reparameterization import Reparameterization

def _norm(p, dim):
    if dim is None:
        return p.norm()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).norm(dim=0).view(*output_size)
    return _norm(p.transpose(0, dim), 0).transpose(0, dim)

HALF_TYPES = (torch.cuda.HalfTensor, torch.HalfTensor)

class WeightNorm(Reparameterization):
    def compute_weight(self, module=None, name=None):
        if module is None:
            module = self.module
        if name is None:
            name = self.name
        module, name = Reparameterization.get_module_and_name(module, name)
        g = getattr(module, name + '_g')
        v = getattr(module, name + '_v')
        fused_weight_norm = Fused_Weight_Norm.apply
        v = v.contiguous()
        w = fused_weight_norm(v, g, self.dim)
        return w

    def reparameterize(self, name, weight, dim):
        names = [name + '_g', name + '_v']
        params = [Parameter(_norm(weight, dim).data), Parameter(weight.data)]
        return names, params