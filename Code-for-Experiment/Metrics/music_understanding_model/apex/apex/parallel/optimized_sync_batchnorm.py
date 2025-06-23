import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import functional as F
import syncbn
from Code_for_Experiment.Metrics.music_understanding_model.apex.apex.parallel.sync_batchnorm_kernel import SyncBatchnormFunction

class SyncBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, process_group=None, channel_last=False):
        super(SyncBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.process_group = process_group
        self.channel_last = channel_last

    def _specify_process_group(self, process_group):
        self.process_group = process_group

    def _specify_channel_last(self, channel_last):
        self.channel_last = channel_last

    def forward(self, input):
        channel_last = self.channel_last if input.dim() != 2 else True
        if not self.training and self.track_running_stats and not channel_last:
            return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)
        else:
            exponential_average_factor = 0.0
            if self.training and self.track_running_stats:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
            return SyncBatchnormFunction.apply(input, self.weight, self.bias, self.running_mean, self.running_var, self.eps, self.training or not self.track_running_stats, exponential_average_factor, self.process_group, channel_last)