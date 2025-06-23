def get_padding(kernel_size: int, dilation: int = 1):
    return (kernel_size * dilation - dilation) // 2