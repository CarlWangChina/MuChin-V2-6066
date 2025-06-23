def get_padding(kernel_size: int, dilation: int = 1):
    return int((kernel_size * dilation - dilation) / 2)