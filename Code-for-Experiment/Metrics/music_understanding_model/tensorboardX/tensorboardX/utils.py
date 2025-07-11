def figure_to_image(figures, close=True):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_agg as plt_backend_agg
        import numpy as np
    except ModuleNotFoundError:
        print('please install matplotlib')

    def render_to_rgb(figure):
        canvas = plt_backend_agg.FigureCanvasAgg(figure)
        canvas.draw()
        data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        w, h = figure.canvas.get_width_height()
        image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
        image_chw = np.moveaxis(image_hwc, source=2, destination=0)
        if close:
            plt.close(figure)
        return image_chw

    if isinstance(figures, list):
        images = [render_to_rgb(figure) for figure in figures]
        return np.stack(images)
    else:
        image = render_to_rgb(figures)
        return image

def graphviz_to_image():
    pass

def _prepare_video(V):
    import numpy as np
    b, t, c, h, w = V.shape
    if V.dtype == np.uint8:
        V = np.float32(V) / 255.

    def is_power2(num):
        return num != 0 and ((num & (num - 1)) == 0)

    if not is_power2(V.shape[0]):
        len_addition = int(2**V.shape[0].bit_length() - V.shape[0])
        V = np.concatenate((V, np.zeros(shape=(len_addition, t, c, h, w))), axis=0)

    n_rows = 2**((b.bit_length() - 1) // 2)
    n_cols = V.shape[0] // n_rows
    V = np.reshape(V, newshape=(n_rows, n_cols, t, c, h, w))
    V = np.transpose(V, axes=(2, 0, 4, 1, 5, 3))
    V = np.reshape(V, newshape=(t, n_rows * h, n_cols * w, c))
    return V

def make_grid(I, ncols=8):
    import numpy as np
    assert isinstance(I, np.ndarray), 'plugin error, should pass numpy array here'
    if I.shape[1] == 1:
        I = np.concatenate([I, I, I], 1)
    assert I.ndim == 4 and I.shape[1] == 3 or I.shape[1] == 4
    nimg = I.shape[0]
    H = I.shape[2]
    W = I.shape[3]
    ncols = min(nimg, ncols)
    nrows = int(np.ceil(float(nimg) / ncols))
    canvas = np.zeros((I.shape[1], H * nrows, W * ncols))
    i = 0
    for y in range(nrows):
        for x in range(ncols):
            if i >= nimg:
                break
            canvas[:, y * H:(y + 1) * H, x * W:(x + 1) * W] = I[i]
            i = i + 1
    return canvas

def convert_to_HWC(tensor, input_format):
    import numpy as np
    assert len(set(input_format)) == len(input_format), "You can not use the same dimension shordhand twice. input_format: {}".format(input_format)
    assert len(tensor.shape) == len(input_format), "size of input tensor and input format are different. tensor shape: {}, input_format: {}".format(tensor.shape, input_format)
    input_format = input_format.upper()
    if len(input_format) == 4:
        index = [input_format.find(c) for c in 'NCHW']
        tensor_NCHW = tensor.transpose(index)
        tensor_CHW = make_grid(tensor_NCHW)
        return tensor_CHW.transpose(1, 2, 0)
    if len(input_format) == 3:
        index = [input_format.find(c) for c in 'HWC']
        tensor_HWC = tensor.transpose(index)
        if tensor_HWC.shape[2] == 1:
            tensor_HWC = np.concatenate([tensor_HWC, tensor_HWC, tensor_HWC], 2)
        return tensor_HWC
    if len(input_format) == 2:
        index = [input_format.find(c) for c in 'HW']
        tensor = tensor.transpose(index)
        tensor = np.stack([tensor, tensor, tensor], 2)
        return tensor