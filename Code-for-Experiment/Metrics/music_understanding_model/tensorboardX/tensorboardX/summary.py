from __future__ import absolute_import, division, print_function
import logging
import numpy as np
import os
import re as _re
from six.moves import range
from .proto.summary_pb2 import Summary
from .proto.summary_pb2 import HistogramProto
from .proto.summary_pb2 import SummaryMetadata
from .proto.tensor_pb2 import TensorProto
from .proto.tensor_shape_pb2 import TensorShapeProto
from .proto.plugin_pr_curve_pb2 import PrCurvePluginData
from .proto.plugin_text_pb2 import TextPluginData
from .proto.plugin_mesh_pb2 import MeshPluginData
from .proto import layout_pb2
from Code_for_Experiment.Metrics.music_understanding_model.tensorboardX.tensorboardX.x2num import make_np
from .utils import _prepare_video, convert_to_HWC
INVALID_TAG_CHARACTERS = _re.compile(r'[^-/\w\.]')

def _clean_tag(name):
    if name is not None:
        new_name = INVALID_TAG_CHARACTERS.sub('_', name)
        new_name = new_name.lstrip('/')
        if new_name != name:
            logging.info('Summary name %s is illegal; using %s instead.' % (name, new_name))
            name = new_name
    return name

def _draw_single_box(image, xmin, ymin, xmax, ymax, display_str, color='black', color_text='black', thickness=2):
    from PIL import ImageDraw, ImageFont
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)
    if display_str:
        text_bottom = bottom
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)], fill=color)
        draw.text((left + margin, text_bottom - text_height - margin), display_str, fill=color_text, font=font)
    return image

def hparams(hparam_dict=None, metric_dict=None):
    from tensorboardX.proto.plugin_hparams_pb2 import HParamsPluginData, SessionEndInfo, SessionStartInfo
    from tensorboardX.proto.api_pb2 import Experiment, HParamInfo, MetricInfo, MetricName, Status
    from six import string_types
    PLUGIN_NAME = 'hparams'
    PLUGIN_DATA_VERSION = 0
    EXPERIMENT_TAG = '_hparams_/experiment'
    SESSION_START_INFO_TAG = '_hparams_/session_start_info'
    SESSION_END_INFO_TAG = '_hparams_/session_end_info'
    hps = [HParamInfo(name=k) for k in hparam_dict.keys()]
    mts = [MetricInfo(name=MetricName(tag=k)) for k in metric_dict.keys()]
    exp = Experiment(hparam_infos=hps, metric_infos=mts)
    content = HParamsPluginData(experiment=exp, version=PLUGIN_DATA_VERSION)
    smd = SummaryMetadata(plugin_data=SummaryMetadata.PluginData(plugin_name=PLUGIN_NAME, content=content.SerializeToString()))
    exp = Summary(value=[Summary.Value(tag=EXPERIMENT_TAG, metadata=smd)])
    ssi = SessionStartInfo()
    for k, v in hparam_dict.items():
        if isinstance(v, string_types):
            ssi.hparams[k].string_value = v
            continue
        if isinstance(v, bool):
            ssi.hparams[k].bool_value = v
            continue
        if not isinstance(v, (int, float)):
            v = make_np(v)[0]
            ssi.hparams[k].number_value = v
    content = HParamsPluginData(session_start_info=ssi, version=PLUGIN_DATA_VERSION)
    smd = SummaryMetadata(plugin_data=SummaryMetadata.PluginData(plugin_name=PLUGIN_NAME, content=content.SerializeToString()))
    ssi = Summary(value=[Summary.Value(tag=SESSION_START_INFO_TAG, metadata=smd)])
    sei = SessionEndInfo(status=Status.STATUS_SUCCESS)
    content = HParamsPluginData(session_end_info=sei, version=PLUGIN_DATA_VERSION)
    smd = SummaryMetadata(plugin_data=SummaryMetadata.PluginData(plugin_name=PLUGIN_NAME, content=content.SerializeToString()))
    sei = Summary(value=[Summary.Value(tag=SESSION_END_INFO_TAG, metadata=smd)])
    return exp, ssi, sei

def scalar(name, scalar, collections=None):
    name = _clean_tag(name)
    scalar = make_np(scalar)
    assert(scalar.squeeze().ndim == 0), 'scalar should be 0D'
    scalar = float(scalar)
    return Summary(value=[Summary.Value(tag=name, simple_value=scalar)])

def histogram_raw(name, min, max, num, sum, sum_squares, bucket_limits, bucket_counts):
    hist = HistogramProto(min=min, max=max, num=num, sum=sum, sum_squares=sum_squares, bucket_limit=bucket_limits, bucket=bucket_counts)
    return Summary(value=[Summary.Value(tag=name, histo=hist)])

def histogram(name, values, bins, max_bins=None):
    name = _clean_tag(name)
    values = make_np(values)
    hist = make_histogram(values.astype(float), bins, max_bins)
    return Summary(value=[Summary.Value(tag=name, histo=hist)])

def make_histogram(values, bins, max_bins=None):
    if values.size == 0:
        raise ValueError('The input has no element.')
    values = values.reshape(-1)
    counts, limits = np.histogram(values, bins=bins)
    num_bins = len(counts)
    if max_bins is not None and num_bins > max_bins:
        subsampling = num_bins // max_bins
        subsampling_remainder = num_bins % subsampling
        if subsampling_remainder != 0:
            counts = np.pad(counts, pad_width=[[0, subsampling - subsampling_remainder]], mode="constant", constant_values=0)
        counts = counts.reshape(-1, subsampling).sum(axis=-1)
        new_limits = np.empty((counts.size + 1,), limits.dtype)
        new_limits[:-1] = limits[:-1:subsampling]
        new_limits[-1] = limits[-1]
        limits = new_limits
    cum_counts = np.cumsum(np.greater(counts, 0, dtype=np.int32))
    start, end = np.searchsorted(cum_counts, [0, cum_counts[-1] - 1], side="right")
    start = int(start)
    end = int(end) + 1
    del cum_counts
    counts = counts[start - 1:end] if start > 0 else np.concatenate([[0], counts[:end]])
    limits = limits[start:end + 1]
    if counts.size == 0 or limits.size == 0:
        raise ValueError('The histogram is empty, please file a bug report.')
    sum_sq = values.dot(values)
    return HistogramProto(min=values.min(), max=values.max(), num=len(values), sum=values.sum(), sum_squares=sum_sq, bucket_limit=limits.tolist(), bucket=counts.tolist())

def image(tag, tensor, rescale=1, dataformats='CHW'):
    tag = _clean_tag(tag)
    tensor = make_np(tensor)
    tensor = convert_to_HWC(tensor, dataformats)
    if tensor.dtype != np.uint8:
        tensor = (tensor * 255.0).astype(np.uint8)
    image = make_image(tensor, rescale=rescale)
    return Summary(value=[Summary.Value(tag=tag, image=image)])

def image_boxes(tag, tensor_image, tensor_boxes, rescale=1, dataformats='CHW', labels=None):
    tensor_image = make_np(tensor_image)
    tensor_image = convert_to_HWC(tensor_image, dataformats)
    tensor_boxes = make_np(tensor_boxes)
    if tensor_image.dtype != np.uint8:
        tensor_image = (tensor_image * 255.0).astype(np.uint8)
    image = make_image(tensor_image, rescale=rescale, rois=tensor_boxes, labels=labels)
    return Summary(value=[Summary.Value(tag=tag, image=image)])

def draw_boxes(disp_image, boxes, labels=None):
    num_boxes = boxes.shape[0]
    list_gt = range(num_boxes)
    for i in list_gt:
        disp_image = _draw_single_box(disp_image, boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3], display_str=None if labels is None else labels[i], color='Red')
    return disp_image

def make_image(tensor, rescale=1, rois=None, labels=None):
    from PIL import Image
    height, width, channel = tensor.shape
    scaled_height = int(height * rescale)
    scaled_width = int(width * rescale)
    image = Image.fromarray(tensor)
    if rois is not None:
        image = draw_boxes(image, rois, labels=labels)
    image = image.resize((scaled_width, scaled_height), Image.ANTIALIAS)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return Summary.Image(height=height, width=width, colorspace=channel, encoded_image_string=image_string)

def video(tag, tensor, fps=4):
    tag = _clean_tag(tag)
    tensor = make_np(tensor)
    tensor = _prepare_video(tensor)
    if tensor.dtype != np.uint8:
        tensor = (tensor * 255.0).astype(np.uint8)
    video = make_video(tensor, fps)
    return Summary(value=[Summary.Value(tag=tag, image=video)])

def make_video(tensor, fps):
    try:
        import moviepy
    except ImportError:
        print('add_video needs package moviepy')
        return
    try:
        from moviepy import editor as mpy
    except ImportError:
        print("moviepy is installed, but can't import moviepy.editor.", "Some packages could be missing [imageio, requests]")
        return
    import tempfile
    t, h, w, c = tensor.shape
    clip = mpy.ImageSequenceClip(list(tensor), fps=fps)
    filename = tempfile.NamedTemporaryFile(suffix='.gif', delete=False).name
    try:
        clip.write_gif(filename, verbose=False, progress_bar=False)
    except TypeError:
        clip.write_gif(filename, verbose=False)
    with open(filename, 'rb') as f:
        tensor_string = f.read()
    try:
        os.remove(filename)
    except OSError:
        logging.warning('The temporary file used by moviepy cannot be deleted.')
    return Summary.Image(height=h, width=w, colorspace=c, encoded_image_string=tensor_string)

def audio(tag, tensor, sample_rate=44100):
    tensor = make_np(tensor)
    if abs(tensor).max() > 1:
        print('warning: audio amplitude out of range, auto clipped.')
        tensor = tensor.clip(-1, 1)
    assert(tensor.ndim == 2), 'input tensor should be 2 dimensional.'
    length_frames, num_channels = tensor.shape
    assert num_channels == 1 or num_channels == 2, f'Expected 1/2 channels, got {num_channels}'
    import soundfile
    import io
    with io.BytesIO() as fio:
        soundfile.write(fio, tensor, samplerate=sample_rate, format='wav')
        audio_string = fio.getvalue()
    audio = Summary.Audio(sample_rate=sample_rate, num_channels=num_channels, length_frames=length_frames, encoded_audio_string=audio_string, content_type='audio/wav')
    return Summary(value=[Summary.Value(tag=tag, audio=audio)])

def custom_scalars(layout):
    categoriesnames = layout.keys()
    categories = []
    layouts = []
    for k, v in layout.items():
        charts = []
        for chart_name, chart_meatadata in v.items():
            tags = chart_meatadata[1]
            if chart_meatadata[0] == 'Margin':
                assert len(tags) == 3
                mgcc = layout_pb2.MarginChartContent(series=[layout_pb2.MarginChartContent.Series(value=tags[0], lower=tags[1], upper=tags[2])])
                chart = layout_pb2.Chart(title=chart_name, margin=mgcc)
            else:
                mlcc = layout_pb2.MultilineChartContent(tag=tags)
                chart = layout_pb2.Chart(title=chart_name, multiline=mlcc)
            charts.append(chart)
        categories.append(layout_pb2.Category(title=k, chart=charts))
    layout = layout_pb2.Layout(category=categories)
    PluginData = SummaryMetadata.PluginData(plugin_name='custom_scalars')
    smd = SummaryMetadata(plugin_data=PluginData)
    tensor = TensorProto(dtype='DT_STRING', string_val=[layout.SerializeToString()], tensor_shape=TensorShapeProto())
    return Summary(value=[Summary.Value(tag='custom_scalars__config__', tensor=tensor, metadata=smd)])

def text(tag, text):
    import json
    PluginData = SummaryMetadata.PluginData(plugin_name='text', content=TextPluginData(version=0).SerializeToString())
    smd = SummaryMetadata(plugin_data=PluginData)
    tensor = TensorProto(dtype='DT_STRING', string_val=[text.encode(encoding='utf_8')], tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]))
    return Summary(value=[Summary.Value(tag=tag + '/text_summary', metadata=smd, tensor=tensor)])

def pr_curve_raw(tag, tp, fp, tn, fn, precision, recall, num_thresholds=127, weights=None):
    if num_thresholds > 127:
        num_thresholds = 127
    data = np.stack((tp, fp, tn, fn, precision, recall))
    pr_curve_plugin_data = PrCurvePluginData(version=0, num_thresholds=num_thresholds).SerializeToString()
    PluginData = SummaryMetadata.PluginData(plugin_name='pr_curves', content=pr_curve_plugin_data)
    smd = SummaryMetadata(plugin_data=PluginData)
    tensor = TensorProto(dtype='DT_FLOAT', float_val=data.reshape(-1).tolist(), tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=data.shape[0]), TensorShapeProto.Dim(size=data.shape[1])]))
    return Summary(value=[Summary.Value(tag=tag, metadata=smd, tensor=tensor)])

def pr_curve(tag, labels, predictions, num_thresholds=127, weights=None):
    num_thresholds = min(num_thresholds, 127)
    data = compute_curve(labels, predictions, num_thresholds=num_thresholds, weights=weights)
    pr_curve_plugin_data = PrCurvePluginData(version=0, num_thresholds=num_thresholds).SerializeToString()
    PluginData = SummaryMetadata.PluginData(plugin_name='pr_curves', content=pr_curve_plugin_data)
    smd = SummaryMetadata(plugin_data=PluginData)
    tensor = TensorProto(dtype='DT_FLOAT', float_val=data.reshape(-1).tolist(), tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=data.shape[0]), TensorShapeProto.Dim(size=data.shape[1])]))
    return Summary(value=[Summary.Value(tag=tag, metadata=smd, tensor=tensor)])

def compute_curve(labels, predictions, num_thresholds=None, weights=None):
    _MINIMUM_COUNT = 1e-7
    if weights is None:
        weights = 1.0
    bucket_indices = np.int32(np.floor(predictions * (num_thresholds - 1)))
    float_labels = labels.astype(np.float)
    histogram_range = (0, num_thresholds - 1)
    tp_buckets, _ = np.histogram(bucket_indices, bins=num_thresholds, range=histogram_range, weights=float_labels * weights)
    fp_buckets, _ = np.histogram(bucket_indices, bins=num_thresholds, range=histogram_range, weights=(1.0 - float_labels) * weights)
    tp = np.cumsum(tp_buckets[::-1])[::-1]
    fp = np.cumsum(fp_buckets[::-1])[::-1]
    tn = fp[0] - fp
    fn = tp[0] - tp
    precision = tp / np.maximum(_MINIMUM_COUNT, tp + fp)
    recall = tp / np.maximum(_MINIMUM_COUNT, tp + fn)
    return np.stack((tp, fp, tn, fn, precision, recall))

def _get_tensor_summary(tag, tensor, content_type, json_config):
    mesh_plugin_data = MeshPluginData(version=0, name=tag, content_type=content_type, json_config=json_config, shape=tensor.shape)
    content = mesh_plugin_data.SerializeToString()
    smd = SummaryMetadata(plugin_data=SummaryMetadata.PluginData(plugin_name='mesh', content=content))
    tensor = TensorProto(dtype='DT_FLOAT', float_val=tensor.reshape(-1).tolist(), tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=tensor.shape[0]), TensorShapeProto.Dim(size=tensor.shape[1]), TensorShapeProto.Dim(size=tensor.shape[2])]))
    tensor_summary = Summary.Value(tag='{}_{}'.format(tag, content_type), tensor=tensor, metadata=smd)
    return tensor_summary

def mesh(tag, vertices, colors, faces, config_dict=None):
    import json
    summaries = []
    tensors = [(vertices, 1), (faces, 2), (colors, 3)]
    for tensor, content_type in tensors:
        if tensor is None:
            continue
        summaries.append(_get_tensor_summary(tag, make_np(tensor), content_type, json.dumps(config_dict, sort_keys=True)))
    return Summary(value=summaries)