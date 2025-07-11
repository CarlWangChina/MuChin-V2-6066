import torch
from torch.autograd.function import Function
import syncbn
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.dist_wrapper import ReduceOp

class SyncBatchnormFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, running_mean, running_variance, eps, track_running_stats=True, momentum=1.0, process_group=None, channel_last=False):
        torch.cuda.nvtx.range_push("sync_BN_fw")
        input = input.contiguous()
        world_size = 0
        mean = None
        var_biased = None
        inv_std = None
        var = None
        out = None
        count = None
        if track_running_stats:
            if channel_last:
                count = int(input.numel()/input.size(-1))
                mean, var_biased = syncbn.welford_mean_var_c_last(input)
            else:
                count = int(input.numel()/input.size(1))
                mean, var_biased = syncbn.welford_mean_var(input)
            if torch.distributed.is_initialized():
                if not process_group:
                    process_group = torch.distributed.group.WORLD
                world_size = torch.distributed.get_world_size(process_group)
                mean_all = torch.empty(world_size, mean.size(0), dtype=mean.dtype, device=mean.device)
                var_all = torch.empty(world_size, var_biased.size(0), dtype=var_biased.dtype, device=var_biased.device)
                mean_l = [mean_all.narrow(0, i, 1) for i in range(world_size)]
                var_l = [var_all.narrow(0, i, 1) for i in range(world_size)]
                torch.distributed.all_gather(mean_l, mean, process_group)
                torch.distributed.all_gather(var_l, var_biased, process_group)
                mean, var, inv_std = syncbn.welford_parallel(mean_all, var_all, count, eps)
            else:
                inv_std = 1.0 / torch.sqrt(var_biased + eps)
                var = var_biased * (count) / (count-1)
            if count == 1 and world_size < 2:
                raise ValueError('Expected more than 1 value per channel when training, got input size{}'.format(input.size()))
            r_m_inc = mean if running_mean.dtype != torch.float16 else mean.half()
            r_v_inc = var if running_variance.dtype != torch.float16 else var.half()
            running_mean.data = running_mean.data * (1-momentum) + momentum*r_m_inc
            running_variance.data = running_variance.data * (1-momentum) + momentum*r_v_inc
        else:
            mean = running_mean.data
            inv_std = 1.0 / torch.sqrt(running_variance.data + eps)
        ctx.save_for_backward(input, weight, mean, inv_std)
        ctx.process_group = process_group
        ctx.channel_last = channel_last
        ctx.world_size = world_size
        if channel_last:
            out = syncbn.batchnorm_forward_c_last(input, mean, inv_std, weight, bias)
        else:
            out = syncbn.batchnorm_forward(input, mean, inv_std, weight, bias)
        torch.cuda.nvtx.range_pop()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        torch.cuda.nvtx.range_push("sync_BN_bw")
        saved_input, weight, mean, inv_std = ctx.saved_tensors
        process_group = ctx.process_group
        channel_last = ctx.channel_last
        world_size = ctx.world_size
        grad_input = grad_weight = grad_bias = None
        if channel_last:
            mean_dy, mean_dy_xmu, grad_weight, grad_bias = syncbn.reduce_bn_c_last(grad_output, saved_input, mean, inv_std, weight)
        else:
            mean_dy, mean_dy_xmu, grad_weight, grad_bias = syncbn.reduce_bn(grad_output, saved_input, mean, inv_std, weight)
        if ctx.needs_input_grad[0]:
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(mean_dy, ReduceOp.SUM, process_group)
                mean_dy = mean_dy / world_size
                torch.distributed.all_reduce(mean_dy_xmu, ReduceOp.SUM, process_group)
                mean_dy_xmu = mean_dy_xmu / world_size
            if channel_last:
                grad_input = syncbn.batchnorm_backward_c_last(grad_output, saved_input, mean, inv_std, weight, mean_dy, mean_dy_xmu)
            else:
                grad_input = syncbn.batchnorm_backward(grad_output, saved_input, mean, inv_std, weight, mean_dy, mean_dy_xmu)
        if weight is None or not ctx.needs_input_grad[1]:
            grad_weight = None
        if weight is None or not ctx.needs_input_grad[2]:
            grad_bias = None
        torch.cuda.nvtx.range_pop()
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None