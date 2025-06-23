import math
import torch
from typing import IO, List, Any, Optional
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.modules.ac.binary import BitUnpacker

class ArithmeticDecoder:
    def __init__(self, fo: IO[bytes], total_range_bits: int = 24):
        self.total_range_bits = total_range_bits
        self.low: int = 0
        self.high: int = 0
        self.current: int = 0
        self.max_bit: int = -1
        self.unpacker = BitUnpacker(bits=1, fo=fo)
        self._dbg: List[Any] = []
        self._dbg2: List[Any] = []
        self._last: Any = None

    @property
    def delta(self) -> int:
        return self.high - self.low + 1

    def _flush_common_prefix(self):
        while self.max_bit >= 0:
            b1 = self.low >> self.max_bit
            b2 = self.high >> self.max_bit
            if b1 == b2:
                self.low -= (b1 << self.max_bit)
                self.high -= (b1 << self.max_bit)
                self.current -= (b1 << self.max_bit)
                assert self.high >= self.low
                assert self.low >= 0
                self.max_bit -= 1
            else:
                break

    def pull(self, quantized_cdf: torch.Tensor) -> Optional[int]:
        while self.delta < 2 ** self.total_range_bits:
            bit = self.unpacker.pull()
            if bit is None:
                return None
            self.low *= 2
            self.high = self.high * 2 + 1
            self.current = self.current * 2 + bit
            self.max_bit += 1

    def bin_search(self, low_idx: int, high_idx: int) -> (int, int, int, int):
        if high_idx < low_idx:
            raise RuntimeError("Binary search failed")
        mid = (low_idx + high_idx) // 2
        range_low = quantized_cdf[mid - 1].item() if mid > 0 else 0
        range_high = quantized_cdf[mid].item() - 1
        effective_low = int(math.ceil(range_low * (self.delta / (2 ** self.total_range_bits))))
        effective_high = int(math.floor(range_high * (self.delta / (2 ** self.total_range_bits))))
        low = effective_low + self.low
        high = effective_high + self.low
        if self.current >= low:
            if self.current <= high:
                return mid, low, high, self.current
            else:
                return self.bin_search(mid + 1, high_idx)
        else:
            return self.bin_search(low_idx, mid - 1)

    def decode(self, quantized_cdf: torch.Tensor) -> Optional[int]:
        self._last = (self.low, self.high, self.current, self.max_bit)
        sym, self.low, self.high, self.current = self.bin_search(0, len(quantized_cdf) - 1)
        self._dbg.append((self.low, self.high, self.current))
        self._flush_common_prefix()
        self._dbg2.append((self.low, self.high, self.current))
        return sym