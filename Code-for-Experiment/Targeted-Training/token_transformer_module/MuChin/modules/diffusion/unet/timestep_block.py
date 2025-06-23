from abc import ABC
import torch.nn as nn
import AttentionBlock
import TransformerBlock

from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.modules.transformers.transformer_blocks import TransformerBlock

class TimestepBlock(ABC, nn.Module):
    def __init__(self):
        super(TimestepBlock, self).__init__()

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, context=None, mask=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, AttentionBlock) or isinstance(layer, TransformerBlock):
                x = layer(x, context, mask=mask)
            else:
                x = layer(x)
        return x