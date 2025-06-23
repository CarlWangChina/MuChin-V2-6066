import gc
import numpy as np
import math
import json
import time
from .summary import compute_curve
from .utils import figure_to_image
from Code_for_Experiment.Metrics.music_understanding_model.tensorboardX.tensorboardX.x2num import make_np

def _check_connection(fn):
    def wrapper(self, *args, **kwargs):
        if not self.server_connected:
            print('ERROR: No Visdom server currently connected')
            self._try_connect()
            return
        fn(self, *args, **kwargs)
    return wrapper

class VisdomWriter:
    def __init__(self, *args, **kwargs):
        try:
            from visdom import Visdom
        except ImportError:
            raise ImportError("Visdom visualization requires installation of Visdom")
        self.scalar_dict = {}
        self.server_connected = False
        self.vis = Visdom(*args, **kwargs)
        self.windows = {}
        self._try_connect()

    def _try_connect(self):
        startup_sec = 1
        self.server_connected = self.vis.check_connection()
        while not self.server_connected and startup_sec > 0:
            time.sleep(0.1)
            startup_sec -= 0.1
            self.server_connected = self.vis.check_connection()
        assert self.server_connected, 'No connection could be formed quickly'

    @_check_connection
    def add_scalar(self, tag, scalar_value, global_step=None, main_tag='default'):
        if self.scalar_dict.get(main_tag) is None:
            self.scalar_dict[main_tag] = {}
        exists = self.scalar_dict[main_tag].get(tag) is not None
        self.scalar_dict[main_tag][tag] = self.scalar_dict[main_tag][tag] + [scalar_value] if exists else [scalar_value]
        plot_name = '{}-{}'.format(main_tag, tag)
        x_val = len(self.scalar_dict[main_tag][tag]) if not global_step else global_step
        if exists:
            self.vis.line(
                X=make_np(x_val),
                Y=make_np(scalar_value),
                name=plot_name,
                update='append',
                win=self.windows[plot_name],
            )
        else:
            self.windows[plot_name] = self.vis.line(
                X=make_np(x_val),
                Y=make_np(scalar_value),
                name=plot_name,
                opts={
                    'title': plot_name,
                    'xlabel': 'timestep',
                    'ylabel': tag,
                },
            )

    @_check_connection
    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None):
        for key in tag_scalar_dict.keys():
            self.add_scalar(key, tag_scalar_dict[key], global_step, main_tag)

    @_check_connection
    def export_scalars_to_json(self, path):
        with open(path, "w") as f:
            json.dump(self.scalar_dict, f)
        self.scalar_dict = {}

    @_check_connection
    def add_histogram(self, tag, values, global_step=None, bins='tensorflow'):
        values = make_np(values)
        self.vis.histogram(make_np(values), opts={'title': tag})

    @_check_connection
    def add_image(self, tag, img_tensor, global_step=None, caption=None):
        img_tensor = make_np(img_tensor)
        self.vis.image(img_tensor, opts={'title': tag, 'caption': caption})

    @_check_connection
    def add_figure(self, tag, figure, global_step=None, close=True):
        self.add_image(tag, figure_to_image(figure, close), global_step)

    @_check_connection
    def add_video(self, tag, vid_tensor, global_step=None, fps=4):
        shape = vid_tensor.shape
        if len(shape) > 4:
            for i in range(shape[0]):
                if isinstance(vid_tensor, np.ndarray):
                    import torch
                    ind_vid = torch.from_numpy(vid_tensor[i, :, :, :, :]).permute(1, 2, 3, 0)
                else:
                    ind_vid = vid_tensor[i, :, :, :, :].permute(1, 2, 3, 0)
                scale_factor = 255 if np.any((ind_vid > 0) & (ind_vid < 1)) else 1
                ind_vid = ind_vid.numpy()
                ind_vid = (ind_vid * scale_factor).astype(np.uint8)
                assert ind_vid.shape[3] in [1, 3, 4], 'Visdom requires the last dimension to be color, which can be 1 (grayscale), 3 (RGB) or 4 (RGBA)'
                self.vis.video(tensor=ind_vid, opts={'fps': fps})
        else:
            self.vis.video(tensor=vid_tensor, opts={'fps': fps})

    @_check_connection
    def add_audio(self, tag, snd_tensor, global_step=None, sample_rate=44100):
        snd_tensor = make_np(snd_tensor)
        self.vis.audio(tensor=snd_tensor, opts={'sample_frequency': sample_rate})

    @_check_connection
    def add_text(self, tag, text_string, global_step=None):
        if text_string is None:
            text_string = tag
        self.vis.text(text_string)

    @_check_connection
    def add_onnx_graph(self, prototxt):
        return

    @_check_connection
    def add_graph(self, model, input_to_model=None, verbose=False, **kwargs):
        return

    @_check_connection
    def add_embedding(self, mat, metadata=None, label_img=None, global_step=None, tag='default', metadata_header=None):
        return

    @_check_connection
    def add_pr_curve(self, tag, labels, predictions, global_step=None, num_thresholds=127, weights=None):
        labels, predictions = make_np(labels), make_np(predictions)
        raw_data = compute_curve(labels, predictions, num_thresholds, weights)
        precision, recall = raw_data[4, :], raw_data[5, :]
        self.vis.line(
            X=recall,
            Y=precision,
            name=tag,
            opts={
                'title': 'PR Curve for {}'.format(tag),
                'xlabel': 'recall',
                'ylabel': 'precision',
            },
        )

    @_check_connection
    def add_pr_curve_raw(self, tag, true_positive_counts, false_positive_counts, true_negative_counts, false_negative_counts, precision, recall, global_step=None, num_thresholds=127, weights=None):
        precision, recall = make_np(precision), make_np(recall)
        self.vis.line(
            X=recall,
            Y=precision,
            name=tag,
            opts={
                'title': 'PR Curve for {}'.format(tag),
                'xlabel': 'recall',
                'ylabel': 'precision',
            },
        )

    def close(self):
        del self.vis
        del self.scalar_dict
        gc.collect()