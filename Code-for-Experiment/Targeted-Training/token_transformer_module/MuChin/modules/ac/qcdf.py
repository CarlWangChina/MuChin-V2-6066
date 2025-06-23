import torch

def build_stable_quantized_cdf(pdf: torch.Tensor, total_range_bits: int, round_off: float = 1e-8, min_range: int = 2, check: bool = True) -> torch.Tensor:
    pdf = pdf.detach()
    if round_off:
        pdf = (pdf / round_off).floor() * round_off
    total_range = 2 ** total_range_bits
    cardinality = len(pdf)
    alpha = min_range * cardinality / total_range
    assert alpha <= 1, "you must reduce min_range"
    ranges = (((1 - alpha) * total_range) * pdf).floor().long()
    ranges += min_range
    quantized_cdf = torch.cumsum(ranges, dim=-1)
    if min_range < 2:
        raise ValueError("min_range must be at least 2.")
    if check:
        assert quantized_cdf[-1] <= 2 ** total_range_bits, quantized_cdf[-1]
        if ((quantized_cdf[1:] - quantized_cdf[:-1]) < min_range).any() or quantized_cdf[0] < min_range:
            raise ValueError("You must increase your total_range_bits.")
    return quantized_cdf