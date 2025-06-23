import torch
import Code_for_Experiment.Metrics.music_understanding

class MetricsCollector:
    def __init__(self, *, parallel_enabled: bool = False, local_rank: int = 0, world_size: int = 1, device: str or torch.device = "cpu"):
        self.local_rank = local_rank
        self.world_size = world_size
        self.parallel_enabled = parallel_enabled
        self.device = device
        self.metrics = {}

    def __getitem__(self, key):
        return self.metrics[key]

    def __setitem__(self, key, value):
        if torch.is_tensor(value):
            value = value.to(self.device)
        else:
            value = torch.tensor(value, device=self.device, dtype=torch.float32)
        self.metrics[key] = value

    def __delitem__(self, key):
        del self.metrics[key]

    def __contains__(self, key):
        return key in self.metrics

    def __len__(self):
        return len(self.metrics)

    def __iter__(self):
        return iter(self.metrics)

    def __str__(self):
        return str(self.metrics)

    def __repr__(self):
        return repr(self.metrics)

    def keys(self):
        return self.metrics.keys()

    def values(self):
        return self.metrics.values()

    def items(self):
        return self.metrics.items()

    def to_dict(self):
        return self.metrics

    def from_dict(self, d: dict):
        self.metrics.clear()
        for k, v in d.items():
            if k.startswith("_"):
                continue
            if torch.is_tensor(v):
                self.metrics[k] = v.to(self.device)
            else:
                self.metrics[k] = torch.tensor(v, device=self.device, dtype=torch.float32)
        return self

    def clear(self):
        self.metrics.clear()

    def __iadd__(self, other: dict):
        for k, v in other.items():
            if k.startswith("_"):
                continue
            if torch.is_tensor(v):
                v = v.to(self.device)
            else:
                v = torch.tensor(v, device=self.device, dtype=torch.float32)
            if k in self.metrics:
                self.metrics[k] += v
            else:
                self.metrics[k] = v
        return self

    def __add__(self, other: dict):
        output = MetricsCollector(parallel_enabled=self.parallel_enabled, local_rank=self.local_rank, device=self.device)
        output.metrics = self.metrics.copy()
        output += other
        return output

    def all_reduce_(self):
        if self.parallel_enabled:
            for k, v in self.metrics.items():
                if k.startswith("_"):
                    continue
                if torch.is_tensor(v):
                    dist.all_reduce(v, op=torch.distributed.ReduceOp.SUM)
                    v /= self.world_size
                    self.metrics[k] = v
        return self

    def all_divide_(self, divisor):
        for k, v in self.metrics.items():
            if k.startswith("_"):
                continue
            self.metrics[k] = v / divisor
        return self