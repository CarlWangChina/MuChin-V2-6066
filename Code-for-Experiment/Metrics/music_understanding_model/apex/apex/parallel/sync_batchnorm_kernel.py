import torch
from torch.autograd.function import Function
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.dist_wrapper import ReduceOp

class SyncBatchnormFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, running_mean, running_variance, eps, process_group, world_size):
        torch.cuda.nvtx.range_push("sync_BN_fw")
        c_last_input = input.transpose(1, -1).contiguous().clone()
        ctx.save_for_backward(c_last_input, weight, bias, running_mean, running_variance)
        ctx.eps = eps
        ctx.process_group = process_group
        ctx.world_size = world_size
        c_last_input = (c_last_input - running_mean) / torch.sqrt(running_variance + eps)
        if weight is not None:
            c_last_input = c_last_input * weight
        if bias is not None:
            c_last_input = c_last_input + bias
        torch.cuda.nvtx.range_pop()
        return c_last_input.transpose(1, -1).contiguous().clone()

    @staticmethod
    def backward(ctx, grad_output):
        torch.cuda.nvtx.range_push("sync_BN_bw")
        c_last_input, weight, bias, running_mean, running_variance = ctx.saved_tensors
        eps = ctx.eps
        process_group = ctx.process_group
        world_size = ctx.world_size
        grad_input = grad_weight = grad_bias = None
        num_features = running_mean.size()[0]
        torch.cuda.nvtx.range_push("carilli field")
        c_last_grad = grad_output.transpose(1, -1).contiguous()
        c_grad = c_last_grad.view(-1, num_features).contiguous()
        torch.cuda.nvtx.range_pop()
        if ctx.needs_input_grad[0]:
            mean_dy = c_grad.mean(0)
            mean_dy_xmu = (c_last_grad * (c_last_input - running_mean)).view(-1, num_features).mean(0)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(mean_dy, ReduceOp.SUM, process_group)
                mean_dy = mean_dy / world_size
                torch.distributed.all_reduce(mean_dy_xmu, ReduceOp.SUM, process_group)
                mean_dy_xmu = mean_dy_xmu / world_size
            c_last_grad_input = (c_last_grad - mean_dy - (c_last_input - running_mean) / (running_variance + eps) * mean_dy_xmu) / torch.sqrt(running_variance + eps)
            if weight is not None:
                c_last_grad_input.mul_(weight)
            grad_input = c_last_grad_input.transpose(1, -1).contiguous()
        grad_weight = None
        if weight is not None and ctx.needs_input_grad[1]:
            grad_weight = ((c_last_input - running_mean) / torch.sqrt(running_variance + eps) * c_last_grad).view(-1, num_features).sum(0)
        grad_bias = None
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = c_grad.sum(0)
        torch.cuda.nvtx.range_pop()
        return grad_input, grad_weight, grad_bias, None, None, None, None, None