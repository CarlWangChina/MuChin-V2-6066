import torch
from Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp._initialize import _initialize
from ._amp_state import _amp_state, warn_or_err, maybe_print

class Properties(object):
    def __init__(self):
        self.options = {
            "enabled": False,
            "opt_level": None,
            "cast_model_type": None,
            "patch_torch_functions": False,
            "keep_batchnorm_fp32": None,
            "master_weights": None,
            "loss_scale": 1.0,
        }

    def _update_options_dict(self, new_options):
        for k, v in new_options.items():
            if k in self.options:
                self.options[k] = v
            else:
                raise ValueError("Tried to set unexpected option {}".format(k))

    def __getattr__(self, name):
        if "options" in self.__dict__:
            options = self.__dict__["options"]
            if name in options:
                return options[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name, value):
        if "options" in self.__dict__:
            if name in self.options:
                if name == "cast_model_type":
                    if self.opt_level == "O1" and value is not None:
                        if value is not False:
                            if value is not torch.float32:
                                warn_or_err("O1 inserts casts around Torch functions rather than "
                                            "model weights, so with O1, the model weights themselves "
                                            "should remain FP32. If you wish to cast the model to a "
                                            "different type, use opt_level='O2' or 'O3'. " +
                                            "cast_model_type was {}".format(value))
                    self.options[name] = value
                elif name == "patch_torch_functions":
                    if self.opt_level != "O1" and value:
                        warn_or_err("Currently, patch_torch_functions=True should only be set by "
                                    "selecting opt_level='O1'.")
                    self.options[name] = value
                elif name == "keep_batchnorm_fp32":
                    if self.opt_level == "O1" and value is not None:
                        warn_or_err("With opt_level O1, batchnorm functions are automatically patched "
                                    "to run in FP32, so keep_batchnorm_fp32 should be None." +
                                    " keep_batchnorm_fp32 was {}".format(value))
                    if value == "False":
                        self.options[name] = False
                    elif value == "True":
                        self.options[name] = True
                    else:
                        assert (value is True or value is False or value is None), \
                            "keep_batchnorm_fp32 must be a boolean, the string 'True' or 'False', " \
                            "or None, found keep_batchnorm_fp32={}".format(value)
                        self.options[name] = value
                elif name == "master_weights":
                    if self.opt_level == "O1" and value is not None:
                        warn_or_err("It doesn't make sense to use master_weights with O1. "
                                    "With O1, your model weights themselves should be FP32.")
                    self.options[name] = value
                elif name == "loss_scale":
                    if value == "dynamic":
                        self.options[name] = value
                    else:
                        self.options[name] = float(value)
                else:
                    self.options[name] = value
        else:
            super(Properties, self).__setattr__(name, value)

class O3:
    brief = "O3:  Pure FP16 training."
    more = "Calls .half() on your model, converting the entire model to FP16.\n" \
           "A casting operation is also inserted to cast incoming Tensors to FP16,\n" \
           "so you don't need to change your data pipeline.\n" \
           "This mode is useful for establishing a performance ceiling.\n" \
           "It's also possible training may 'just work' in this mode.\n" \
           "If not, try other optimization levels."

    def __call__(self, properties):
        properties.enabled = True
        properties.opt_level = "O3"
        properties.cast_model_type = torch.float16
        properties.patch_torch_functions = False
        properties.keep_batchnorm_fp32 = False
        properties.master_weights = False
        properties.loss_scale = 1.0
        return properties

class O2:
    brief = "O2:  FP16 training with FP32 batchnorm and FP32 master weights.\n"
    more = "Calls .half() on your model, converting the entire model (except for batchnorms)\n" \
           "to FP16.  Batchnorms are retained in FP32 for additional stability.\n" \
           "The forward pass is patched to cast incoming Tensors to FP16, so you don't need to change\n" \
           "your data pipeline.\n" \
           "O2 creates FP32 master weights outside the model and patches any optimizers to update\n" \
           "these master weights, then copy the master weights into the FP16 model weights.\n" \
           "Master weights can also improve convergence and stability."

    def __call__(self, properties):
        properties.enabled = True
        properties.opt_level = "O2"
        properties.cast_model_type = torch.float16
        properties.patch_torch_functions = False
        properties.keep_batchnorm_fp32 = True
        properties.master_weights = True
        properties.loss_scale = "dynamic"
        return properties

class O1:
    brief = "O1:  Insert automatic casts around Pytorch functions and Tensor methods.\n"
    more = "The type of your model's weights is not altered.  However, internally,\n" \
           "Pytorch functions are patched to cast any Tensor Core-friendly ops to FP16 for speed,\n" \
           "while operations that might benefit from the additional stability of FP32 are patched\n" \
           "to cast their inputs to fp32.\n" \
           "O1 is the safest way to try mixed precision training, and is recommended when\n" \
           "trying mixed precision training for the first time."

    def __call__(self, properties):
        properties.enabled = True
        properties.opt_level = "O1"
        properties.cast_model_type = None
        properties.patch_torch_functions = True
        properties.keep_batchnorm_fp32 = None
        properties.master_weights = None
        properties.loss_scale = "dynamic"
        return properties

class O0:
    brief = "O0:  Pure FP32 training.\n"
    more = "Your models are checked to make sure parameters are FP32, but otherwise the\n" \
           "types of weights and internal Pytorch operations are not altered.  This mode disables any\n" \
           "FP16 arithmetic, although other optimizations like DDP interop may still be requested.\n"

    def __call__(self, properties):
        properties.enabled = True
        properties.opt_level = "O0"
        properties.cast_model_type = torch.float32
        properties.patch_torch_functions = False
        properties.keep_batchnorm_fp32 = None
        properties.master_weights = False
        properties.loss_scale = 1.0
        return properties

opt_levels = {
    "O3": O3(),
    "O2": O2(),
    "O1": O1(),
    "O0": O0()
}

def initialize(
    models,
    optimizers=None,
    enabled=True,
    opt_level="O1",
    cast_model_type=None,
    patch_torch_functions=None,
    keep_batchnorm_fp32=None,
    master_weights=None,
    loss_scale=None,
    cast_model_outputs=None,
    num_losses=1,
    verbosity=1,
    min_loss_scale=None,
    max_loss_scale=2. ** 24
):
    _amp_state.opt_properties = Properties()
    _amp_state.verbosity = verbosity
    if not enabled:
        if optimizers is None:
            return models
        else:
            return models, optimizers
    if not torch.backends.cudnn.enabled:
        raise RuntimeError(
            "Amp requires torch.backends.cudnn.enabled = True")
    if opt_level not in opt_levels:
        raise RuntimeError(
            "Unexpected optimization level {}. ".format(opt_level) +
            "Options are 'O0', 'O1', 'O2', 'O3'.  Note that in `O0`, `O1`, etc., the prefix O is the letter O, " +
            "not the number zero.")
    else:
        _amp_state.opt_properties = opt_levels[opt_level](_amp_state.opt_properties)
        maybe_print("Selected optimization level {}".format(opt_levels[opt_level].brief), True)
        maybe_print("Defaults for this optimization level are:", True)
        for k, v in _amp_state.opt_properties.options.items():
            maybe_print("{:22} : {}".format(k, v), True)
    _amp_state.min_loss_scale = min_loss_scale
    _amp_state.max_loss_scale = max_loss_scale
    maybe_print("Processing user overrides (additional kwargs that are not None)...", True)
    if enabled is not None:
        _amp_state.opt_properties.enabled = enabled
    if opt_level is not None:
        _amp_state.opt_properties.opt_level = opt_level
    if cast_model_type is not None:
        _amp_state.opt_properties.cast_model_type = cast_model_type
    if patch_torch_functions is not None:
        _amp_state.opt_properties.patch_torch_functions = patch_torch_functions
    if keep_batchnorm_fp32 is not None:
        _amp_state.opt_properties.keep_batchnorm_fp32 = keep_batchnorm_fp32
    if master_weights is not None:
        _amp_state.opt_properties.master_weights = master_weights
    if loss_scale is not None:
        _amp_state.opt_properties.loss_scale = loss_scale
    maybe_print("After processing overrides, optimization options are:", True)
    for k, v in _amp_state.opt_properties.options.items():
        maybe_print("{:22} : {}".format(k, v), True)
    return _initialize(models, optimizers, _amp_state.opt_properties, num_losses, cast_model_outputs)