import math
import torch
from typing import IO, List, Any
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.modules.ac.binary import BitPacker

class ArithmeticEncoder:
    def __init__(self, fo: IO[bytes], total_range_bits: int = 24):
        assert total_range_bits <= 30
        self.total_range_bits = total_range_bits
        self.packer = BitPacker(bits=1, fo=fo)
        self.low: int = 0
        self.high: int = 0
        self.max_bit: int = -1
        self._dbg: List[Any] = []
        self._dbg2: List[Any] = []

    @property
    def delta(self) -> int:
        return self.high - self.low + 1

    def _flush_common_prefix(self):
        assert self.high >= self.low, (self.low, self.high)
        assert self.high < 2 ** (self.max_bit + 1)
        while self.max_bit >= 0:
            b1 = self.low >> self.max_bit
            b2 = self.high >> self.max_bit
            if b1 == b2:
                self.low -= (b1 << self.max_bit)
                self.high -= (b1 << self.max_bit)
                assert self.high >= self.low, (self.high, self.low, self.max_bit)
                assert self.low >= 0
                self.max_bit -= 1
                self.packer.push(b1)
            else:
                break

    def push(self, symbol: int, quantized_cdf: torch.Tensor):
        while self.delta < 2 ** self.total_range_bits:
            self.low *= 2
            self.high = self.high * 2 + 1
            self.max_bit += 1
        range_low = 0 if symbol == 0 else quantized_cdf[symbol - 1].item()
        range_high = quantized_cdf[symbol].item() - 1
        effective_low = int(math.ceil(range_low * (self.delta / (2 ** self.total_range_bits))))
        effective_high = int(math.floor(range_high * (self.delta / (2 ** self.total_range_bits))))
        assert self.low <= self.high
        self.high = self.low + effective_high
        self.low = self.low + effective_low
        assert self.low <= self.high, (effective_low, effective_high, range_low, range_high)
        self._dbg.append((self.low, self.high))
        self._dbg2.append((self.low, self.high))
        self._flush_common_prefix()
        assert self.low <= self.high
        assert self.max_bit >= -1
        assert self.max_bit <= 61, self.max_bit

    def flush(self):
        while self.max_bit >= 0:
            b1 = (self.low >> self.max_bit) & 1
            self.packer.push(b1)
            self.max_bit -= 1
        self.packer.flush()