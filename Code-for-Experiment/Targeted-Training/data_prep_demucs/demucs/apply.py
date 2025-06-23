from concurrent.futures import ThreadPoolExecutor
import copy
import random
from threading import Lock
import typing as tp
import torch as th
from torch import nn
from torch.nn import functional as F
import tqdm
from .demucs import Demucs
from .hdemucs import HDemucs
from Code_for_Experiment.Targeted_Training.data_prep_demucs.demucs.htdemucs import HTDemucs
from .utils import center_trim, DummyPoolExecutor

Model = tp.Union[Demucs, HDemucs, HTDemucs]

class BagOfModels(nn.Module):
    def __init__(self, models: tp.List[Model],
                 weights: tp.Optional[tp.List[tp.List[float]]] = None,
                 segment: tp.Optional[float] = None):
        super().__init__()
        assert len(models) > 0
        first = models[0]
        for other in models:
            assert other.sources == first.sources
            assert other.samplerate == first.samplerate
            assert other.audio_channels == first.audio_channels
            if segment is not None:
                if not isinstance(other, HTDemucs) and segment > other.segment:
                    other.segment = segment
        self.audio_channels = first.audio_channels
        self.samplerate = first.samplerate
        self.sources = first.sources
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [[1. for _ in first.sources] for _ in models]
        else:
            assert len(weights) == len(models)
            for weight in weights:
                assert len(weight) == len(first.sources)
        self.weights = weights

    @property
    def max_allowed_segment(self) -> float:
        max_allowed_segment = float('inf')
        for model in self.models:
            if isinstance(model, HTDemucs):
                max_allowed_segment = min(max_allowed_segment, float(model.segment))
        return max_allowed_segment

    def forward(self, x):
        raise NotImplementedError("Call `apply_model` on this.")

class TensorChunk:
    def __init__(self, tensor, offset=0, length=None):
        total_length = tensor.shape[-1]
        assert offset >= 0
        assert offset < total_length
        if length is None:
            length = total_length - offset
        else:
            length = min(total_length - offset, length)
        if isinstance(tensor, TensorChunk):
            self.tensor = tensor.tensor
            self.offset = offset + tensor.offset
        else:
            self.tensor = tensor
            self.offset = offset
        self.length = length
        self.device = tensor.device

    @property
    def shape(self):
        shape = list(self.tensor.shape)
        shape[-1] = self.length
        return shape

    def padded(self, target_length):
        delta = target_length - self.length
        total_length = self.tensor.shape[-1]
        assert delta >= 0
        start = self.offset - delta // 2
        end = start + target_length
        correct_start = max(0, start)
        correct_end = min(total_length, end)
        pad_left = correct_start - start
        pad_right = end - correct_end
        out = F.pad(self.tensor[..., correct_start:correct_end], (pad_left, pad_right))
        assert out.shape[-1] == target_length
        return out

def tensor_chunk(tensor_or_chunk):
    if isinstance(tensor_or_chunk, TensorChunk):
        return tensor_or_chunk
    else:
        assert isinstance(tensor_or_chunk, th.Tensor)
        return TensorChunk(tensor_or_chunk)

def _replace_dict(_dict: tp.Optional[dict], *subs: tp.Tuple[tp.Hashable, tp.Any]) -> dict:
    if _dict is None:
        _dict = {}
    else:
        _dict = copy.copy(_dict)
    for key, value in subs:
        _dict[key] = value
    return _dict

def apply_model(model: tp.Union[BagOfModels, Model],
                mix: tp.Union[th.Tensor, TensorChunk],
                shifts: int = 1, split: bool = True,
                overlap: float = 0.25, transition_power: float = 1.,
                progress: bool = False, device=None,
                num_workers: int = 0, segment: tp.Optional[float] = None,
                pool=None, lock=None,
                callback: tp.Optional[tp.Callable[[dict], None]] = None,
                callback_arg: tp.Optional[dict] = None) -> th.Tensor:
    if device is None:
        device = mix.device
    else:
        device = th.device(device)
    if pool is None:
        if num_workers > 0 and device.type == 'cpu':
            pool = ThreadPoolExecutor(num_workers)
        else:
            pool = DummyPoolExecutor()
    if lock is None:
        lock = Lock()
    callback_arg = _replace_dict(
        callback_arg, *{"model_idx_in_bag": 0, "shift_idx": 0, "segment_offset": 0}.items()
    )
    kwargs: tp.Dict[str, tp.Any] = {
        'shifts': shifts,
        'split': split,
        'overlap': overlap,
        'transition_power': transition_power,
        'progress': progress,
        'device': device,
        'pool': pool,
        'segment': segment,
        'lock': lock,
    }
    out: tp.Union[float, th.Tensor]
    res: tp.Union[float, th.Tensor]
    if isinstance(model, BagOfModels):
        estimates: tp.Union[float, th.Tensor] = 0.
        totals = [0.] * len(model.sources)
        callback_arg["models"] = len(model.models)
        for sub_model, model_weights in zip(model.models, model.weights):
            kwargs["callback"] = (
                (lambda d, i=callback_arg["model_idx_in_bag"]: callback(
                    _replace_dict(d, ("model_idx_in_bag", i))) if callback else None)
            )
            original_model_device = next(iter(sub_model.parameters())).device
            sub_model.to(device)
            res = apply_model(sub_model, mix, **kwargs, callback_arg=callback_arg)
            out = res
            sub_model.to(original_model_device)
            for k, inst_weight in enumerate(model_weights):
                out[:, k, :, :] *= inst_weight
                totals[k] += inst_weight
            estimates += out
            del out
            callback_arg["model_idx_in_bag"] += 1
        assert isinstance(estimates, th.Tensor)
        for k in range(estimates.shape[1]):
            estimates[:, k, :, :] /= totals[k]
        return estimates
    if "models" not in callback_arg:
        callback_arg["models"] = 1
    model.to(device)
    model.eval()
    assert transition_power >= 1, "transition_power < 1 leads to weird behavior."
    batch, channels, length = mix.shape
    if shifts:
        kwargs['shifts'] = 0
        max_shift = int(0.5 * model.samplerate)
        mix = tensor_chunk(mix)
        assert isinstance(mix, TensorChunk)
        padded_mix = mix.padded(length + 2 * max_shift)
        out = 0.
        for shift_idx in range(shifts):
            offset = random.randint(0, max_shift)
            shifted = TensorChunk(padded_mix, offset, length + max_shift - offset)
            kwargs["callback"] = (
                (lambda d, i=shift_idx: callback(_replace_dict(d, ("shift_idx", i)))
                 if callback else None)
            )
            res = apply_model(model, shifted, **kwargs, callback_arg=callback_arg)
            shifted_out = res
            out += shifted_out[..., max_shift - offset:]
        out /= shifts
        assert isinstance(out, th.Tensor)
        return out
    elif split:
        kwargs['split'] = False
        out = th.zeros(batch, len(model.sources), channels, length, device=mix.device)
        sum_weight = th.zeros(length, device=mix.device)
        if segment is None:
            segment = model.segment
        assert segment is not None and segment > 0.
        segment_length: int = int(model.samplerate * segment)
        stride = int((1 - overlap) * segment_length)
        offsets = range(0, length, stride)
        scale = float(format(stride / model.samplerate, ".2f"))
        weight = th.cat([th.arange(1, segment_length // 2 + 1, device=device),
                         th.arange(segment_length - segment_length // 2, 0, -1, device=device)])
        assert len(weight) == segment_length
        weight = (weight / weight.max())**transition_power
        futures = []
        for offset in offsets:
            chunk = TensorChunk(mix, offset, segment_length)
            future = pool.submit(apply_model, model, chunk, **kwargs, callback_arg=callback_arg,
                                 callback=(lambda d, i=offset:
                                           callback(_replace_dict(d, ("segment_offset", i)))
                                           if callback else None))
            futures.append((future, offset))
            offset += segment_length
        if progress:
            futures = tqdm.tqdm(futures, unit_scale=scale, ncols=120, unit='seconds')
        for future, offset in futures:
            try:
                chunk_out = future.result()
            except Exception:
                pool.shutdown(wait=True, cancel_futures=True)
                raise
            chunk_length = chunk_out.shape[-1]
            out[..., offset:offset + segment_length] += (
                weight[:chunk_length] * chunk_out).to(mix.device)
            sum_weight[offset:offset + segment_length] += weight[:chunk_length].to(mix.device)
        assert sum_weight.min() > 0
        out /= sum_weight
        assert isinstance(out, th.Tensor)
        return out
    else:
        valid_length: int
        if isinstance(model, HTDemucs) and segment is not None:
            valid_length = int(segment * model.samplerate)
        elif hasattr(model, 'valid_length'):
            valid_length = model.valid_length(length)
        else:
            valid_length = length
        mix = tensor_chunk(mix)
        assert isinstance(mix, TensorChunk)
        padded_mix = mix.padded(valid_length).to(device)
        with lock:
            if callback is not None:
                callback(_replace_dict(callback_arg, ("state", "start")))
        with th.no_grad():
            out = model(padded_mix)
        with lock:
            if callback is not None:
                callback(_replace_dict(callback_arg, ("state", "end")))
        assert isinstance(out, th.Tensor)
        return center_trim(out, length)