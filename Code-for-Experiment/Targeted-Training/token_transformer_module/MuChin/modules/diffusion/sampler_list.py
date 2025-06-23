import importlib
import torch
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.modules.diffusion.samplers.sampler_base import Sampler

_samplers = {
    "ddim": "ama-prof-divi.modules.diffusion.samplers.ddim.DDIMSampler",
    "ddpm": "ama-prof-divi.modules.diffusion.samplers.ddpm.DDPMSampler",
    "dpm++2m": "ama-prof-divi.modules.diffusion.samplers.dpm2m.DPMSolverMultistepSampler",
    "unipc": "ama-prof-divi.modules.diffusion.samplers.unipc.UniPCMultistepSampler",
    "euler": "ama-prof-divi.modules.diffusion.samplers.euler.EulerSampler",
    "heun": "ama-prof-divi.modules.diffusion.samplers.heun.HeunSampler",
}

def get_sampler(name: str, args: dict = None, *, training: bool = False, device: torch.device or str = "cpu") -> Sampler:
    if name not in _samplers:
        raise NameError("Sampler {} is not found.".format(name))
    cls_name = _samplers[name]
    pkg_name = cls_name[:cls_name.rfind(".")]
    try:
        package = importlib.import_module(pkg_name)
        cls = getattr(package, cls_name[cls_name.rfind(".") + 1:])
        return cls(name=name, args=args if args is not None else {}, training=training, device=device)
    except KeyError:
        raise NameError("Sampler class '{}' is not found.".format(cls_name))

def get_available_samplers() -> [str]:
    return list(_samplers.keys())