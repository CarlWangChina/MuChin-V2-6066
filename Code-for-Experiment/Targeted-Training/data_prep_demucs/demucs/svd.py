import random
import torch

def power_iteration(m, niters=1, bs=1):
    assert m.dim() == 2
    assert m.shape[0] == m.shape[1]
    dim = m.shape[0]
    b = torch.randn(dim, bs, device=m.device, dtype=m.dtype)
    for _ in range(niters):
        n = m.mm(b)
        norm = n.norm(dim=0, keepdim=True)
        b = n / (1e-10 + norm)
    return norm.mean()

penalty_rng = random.Random(1234)

def svd_penalty(model, min_size=0.1, dim=1, niters=2, powm=False, convtr=True, proba=1, conv_only=False, exact=False, bs=1):
    total = 0
    if penalty_rng.random() > proba:
        return 0.0
    for m in model.modules():
        for name, p in m.named_parameters(recurse=False):
            if p.numel() / 2**18 < min_size:
                continue
            if convtr:
                if isinstance(m, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d)):
                    if p.dim() in [3, 4]:
                        p = p.transpose(0, 1).contiguous()
            if p.dim() == 3:
                p = p.view(len(p), -1)
            elif p.dim() == 4:
                p = p.view(len(p), -1)
            elif p.dim() == 1:
                continue
            elif conv_only:
                continue
            assert p.dim() == 2, (name, p.shape)
            if exact:
                estimate = torch.svd(p, compute_uv=False)[1].pow(2).max()
            elif powm:
                a, b = p.shape
                if a < b:
                    n = p.mm(p.t())
                else:
                    n = p.t().mm(p)
                estimate = power_iteration(n, niters, bs)
            else:
                estimate = torch.svd_lowrank(p, dim, niters)[1][0].pow(2)
            total += estimate
    return total / proba