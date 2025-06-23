from __future__ import absolute_import, division, print_function
import json
import os
import six
import time
import logging
from .embedding import make_mat, make_sprite, make_tsv, append_pbtxt
from .event_file_writer import EventFileWriter
from .onnx_graph import load_onnx_graph
from .pytorch_graph import graph
from .proto import event_pb2
from .proto import summary_pb2
from .proto.event_pb2 import SessionLog, Event
from .utils import figure_to_image
from .summary import (scalar, histogram, histogram_raw, image, audio, text,
                      pr_curve, pr_curve_raw, video, custom_scalars, image_boxes, mesh, hparams)

class DummyFileWriter(object):
    def __init__(self, logdir):
        self._logdir = logdir
    def get_logdir(self):
        return self._logdir
    def add_event(self, event, step=None, walltime=None):
        return
    def add_summary(self, summary, global_step=None, walltime=None):
        return
    def add_graph(self, graph_profile, walltime=None):
        return
    def add_onnx_graph(self, graph, walltime=None):
        return
    def flush(self):
        return
    def close(self):
        return
    def reopen(self):
        return

class FileWriter(object):
    def __init__(self, logdir, max_queue=10, flush_secs=120, filename_suffix=''):
        logdir = str(logdir)
        self.event_writer = EventFileWriter(
            logdir, max_queue, flush_secs, filename_suffix)
    def get_logdir(self):
        return self.event_writer.get_logdir()
    def add_event(self, event, step=None, walltime=None):
        event.wall_time = time.time() if walltime is None else walltime
        if step is not None:
            event.step = int(step)
        self.event_writer.add_event(event)
    def add_summary(self, summary, global_step=None, walltime=None):
        event = event_pb2.Event(summary=summary)
        self.add_event(event, global_step, walltime)
    def add_graph(self, graph_profile, walltime=None):
        graph = graph_profile[0]
        stepstats = graph_profile[1]
        event = event_pb2.Event(graph_def=graph.SerializeToString())
        self.add_event(event, None, walltime)
        trm = event_pb2.TaggedRunMetadata(
            tag='step1', run_metadata=stepstats.SerializeToString())
        event = event_pb2.Event(tagged_run_metadata=trm)
        self.add_event(event, None, walltime)
    def add_onnx_graph(self, graph, walltime=None):
        event = event_pb2.Event(graph_def=graph.SerializeToString())
        self.add_event(event, None, walltime)
    def flush(self):
        self.event_writer.flush()
    def close(self):
        self.event_writer.close()
    def reopen(self):
        self.event_writer.reopen()

class SummaryWriter(object):
    def __init__(self, logdir=None, comment='', purge_step=None, max_queue=10,
                 flush_secs=120, filename_suffix='', write_to_disk=True, log_dir=None, **kwargs):
        if log_dir is not None and logdir is None:
            logdir = log_dir
        if not logdir:
            import socket
            from datetime import datetime
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            logdir = os.path.join(
                'runs', current_time + '_' + socket.gethostname() + comment)
        self.logdir = logdir
        self.purge_step = purge_step
        self._max_queue = max_queue
        self._flush_secs = flush_secs
        self._filename_suffix = filename_suffix
        self._write_to_disk = write_to_disk
        self.kwargs = kwargs
        self.file_writer = self.all_writers = None
        self._get_file_writer()
        v = 1E-12
        buckets = []
        neg_buckets = []
        while v < 1E20:
            buckets.append(v)
            neg_buckets.append(-v)
            v *= 1.1
        self.default_bins = neg_buckets[::-1] + [0] + buckets
        self.scalar_dict = {}
    def __append_to_scalar_dict(self, tag, scalar_value, global_step,
                                 timestamp):
        from Code_for_Experiment.Metrics.music_understanding_model.tensorboardX.tensorboardX.x2num import make_np
        if tag not in self.scalar_dict.keys():
            self.scalar_dict[tag] = []
        self.scalar_dict[tag].append(
            [timestamp, global_step, float(make_np(scalar_value))])
    def _check_caffe2_blob(self, item):
        return isinstance(item, six.string_types)
    def _get_file_writer(self):
        if not self._write_to_disk:
            self.file_writer = DummyFileWriter(logdir=self.logdir)
            self.all_writers = {self.file_writer.get_logdir(): self.file_writer}
            return self.file_writer
        if self.all_writers is None or self.file_writer is None:
            if 'purge_step' in self.kwargs.keys():
                most_recent_step = self.kwargs.pop('purge_step')
                self.file_writer = FileWriter(logdir=self.logdir,
                                              max_queue=self._max_queue,
                                              flush_secs=self._flush_secs,
                                              filename_suffix=self._filename_suffix,
                                              **self.kwargs)
                self.file_writer.add_event(
                    Event(step=most_recent_step, file_version='brain.Event:2'))
                self.file_writer.add_event(
                    Event(step=most_recent_step, session_log=SessionLog(status=SessionLog.START)))
            else:
                self.file_writer = FileWriter(logdir=self.logdir,
                                              max_queue=self._max_queue,
                                              flush_secs=self._flush_secs,
                                              filename_suffix=self._filename_suffix,
                                              **self.kwargs)
            self.all_writers = {self.file_writer.get_logdir(): self.file_writer}
        return self.file_writer
    def add_hparams(self, hparam_dict=None, metric_dict=None):
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)
        with SummaryWriter(logdir=os.path.join(self.file_writer.get_logdir(), str(time.time()))) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)
    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        if self._check_caffe2_blob(scalar_value):
            scalar_value = workspace.FetchBlob(scalar_value)
        self._get_file_writer().add_summary(
            scalar(tag, scalar_value), global_step, walltime)
    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        walltime = time.time() if walltime is None else walltime
        fw_logdir = self._get_file_writer().get_logdir()
        for tag, scalar_value in tag_scalar_dict.items():
            fw_tag = fw_logdir + "/" + main_tag + "/" + tag
            if fw_tag in self.all_writers.keys():
                fw = self.all_writers[fw_tag]
            else:
                fw = FileWriter(logdir=fw_tag)
                self.all_writers[fw_tag] = fw
            if self._check_caffe2_blob(scalar_value):
                scalar_value = workspace.FetchBlob(scalar_value)
            fw.add_summary(scalar(main_tag, scalar_value),
                           global_step, walltime)
            self.__append_to_scalar_dict(
                fw_tag, scalar_value, global_step, walltime)
    def export_scalars_to_json(self, path):
        with open(path, "w") as f:
            json.dump(self.scalar_dict, f)
        self.scalar_dict = {}
    def add_histogram(self, tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None):
        if self._check_caffe2_blob(values):
            values = workspace.FetchBlob(values)
        if isinstance(bins, six.string_types) and bins == 'tensorflow':
            bins = self.default_bins
        self._get_file_writer().add_summary(
            histogram(tag, values, bins, max_bins=max_bins), global_step, walltime)
    def add_histogram_raw(self, tag, min, max, num, sum, sum_squares,
                          bucket_limits, bucket_counts, global_step=None,
                          walltime=None):
        if len(bucket_limits) != len(bucket_counts):
            raise ValueError('len(bucket_limits) != len(bucket_counts), see the document.')
        self._get_file_writer().add_summary(
            histogram_raw(tag,
                          min,
                          max,
                          num,
                          sum,
                          sum_squares,
                          bucket_limits,
                          bucket_counts),
            global_step,
            walltime)
    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        if self._check_caffe2_blob(img_tensor):
            img_tensor = workspace.FetchBlob(img_tensor)
        self._get_file_writer().add_summary(
            image(tag, img_tensor, dataformats=dataformats), global_step, walltime)
    def add_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW'):
        if self._check_caffe2_blob(img_tensor):
            img_tensor = workspace.FetchBlob(img_tensor)
        if isinstance(img_tensor, list):
            if dataformats.upper() != 'CHW' and dataformats.upper() != 'HWC':
                print('A list of image is passed, but the dataformat is neither CHW nor HWC.')
                print('Nothing is written.')
                return
            import torch
            try:
                img_tensor = torch.stack(img_tensor, 0)
            except TypeError as e:
                import numpy as np
                img_tensor = np.stack(img_tensor, 0)
            dataformats = 'N' + dataformats
        self._get_file_writer().add_summary(
            image(tag, img_tensor, dataformats=dataformats), global_step, walltime)
    def add_image_with_boxes(self, tag, img_tensor, box_tensor, global_step=None,
                             walltime=None, dataformats='CHW', labels=None, **kwargs):
        if self._check_caffe2_blob(img_tensor):
            img_tensor = workspace.FetchBlob(img_tensor)
        if self._check_caffe2_blob(box_tensor):
            box_tensor = workspace.FetchBlob(box_tensor)
        if labels is not None:
            if isinstance(labels, str):
                labels = [labels]
            if len(labels) != box_tensor.shape[0]:
                logging.warning('Number of labels do not equal to number of box, skip the labels.')
                labels = None
        self._get_file_writer().add_summary(image_boxes(
            tag, img_tensor, box_tensor, dataformats=dataformats, labels=labels, **kwargs), global_step, walltime)
    def add_figure(self, tag, figure, global_step=None, close=True, walltime=None):
        if isinstance(figure, list):
            self.add_image(tag, figure_to_image(figure, close), global_step, walltime, dataformats='NCHW')
        else:
            self.add_image(tag, figure_to_image(figure, close), global_step, walltime, dataformats='CHW')
    def add_video(self, tag, vid_tensor, global_step=None, fps=4, walltime=None):
        self._get_file_writer().add_summary(
            video(tag, vid_tensor, fps), global_step, walltime)
    def add_audio(self, tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None):
        if self._check_caffe2_blob(snd_tensor):
            snd_tensor = workspace.FetchBlob(snd_tensor)
        self._get_file_writer().add_summary(
            audio(tag, snd_tensor, sample_rate=sample_rate), global_step, walltime)
    def add_text(self, tag, text_string, global_step=None, walltime=None):
        self._get_file_writer().add_summary(
            text(tag, text_string), global_step, walltime)
    def add_onnx_graph(self, prototxt):
        self._get_file_writer().add_onnx_graph(load_onnx_graph(prototxt))
    def add_graph(self, model, input_to_model=None, verbose=False, **kwargs):
        if hasattr(model, 'forward'):
            import torch
            from distutils.version import LooseVersion
            if LooseVersion(torch.__version__) >= LooseVersion("0.3.1"):
                pass
            else:
                if LooseVersion(torch.__version__) >= LooseVersion("0.3.0"):
                    print('You are using PyTorch==0.3.0, use add_onnx_graph()')
                    return
                if not hasattr(torch.autograd.Variable, 'grad_fn'):
                    print('add_graph() only supports PyTorch v0.2.')
                    return
            self._get_file_writer().add_graph(graph(model, input_to_model, verbose, **kwargs))
        else:
            from caffe2.proto import caffe2_pb2
            from caffe2.python import core
            from .caffe2_graph import (
                model_to_graph_def, nets_to_graph_def, protos_to_graph_def
            )
            if isinstance(model, list):
                if isinstance(model[0], core.Net):
                    current_graph = nets_to_graph_def(
                        model, **kwargs)
                elif isinstance(model[0], caffe2_pb2.NetDef):
                    current_graph = protos_to_graph_def(
                        model, **kwargs)
            else:
                current_graph = model_to_graph_def(
                    model, **kwargs)
            event = event_pb2.Event(
                graph_def=current_graph.SerializeToString())
            self._get_file_writer().add_event(event)
    @staticmethod
    def _encode(rawstr):
        retval = rawstr
        retval = retval.replace("%", "%%%02x" % (ord("%")))
        retval = retval.replace("/", "%%%02x" % (ord("/")))
        retval = retval.replace("\\", "%%%02x" % (ord("\\")))
        return retval
    def add_embedding(self, mat, metadata=None, label_img=None, global_step=None, tag='default', metadata_header=None):
        from Code_for_Experiment.Metrics.music_understanding_model.tensorboardX.tensorboardX.x2num import make_np
        mat = make_np(mat)
        if global_step is None:
            global_step = 0
        subdir = "%s/%s" % (str(global_step).zfill(5), self._encode(tag))
        save_path = os.path.join(self._get_file_writer().get_logdir(), subdir)
        try:
            os.makedirs(save_path)
        except OSError:
            print(
                'warning: Embedding dir exists, did you set global_step for add_embedding()?')
        if metadata is not None:
            assert mat.shape[0] == len(
                metadata), 'mat.shape[0] != len(metadata)'
            make_tsv(metadata, save_path, metadata_header=metadata_header)
        if label_img is not None:
            assert mat.shape[0] == label_img.shape[0], 'mat.shape[0] != label_img.shape[0]'
            assert label_img.shape[2] == label_img.shape[3], 'Image should be square, see tensorflow/tensorboard'
            make_sprite(label_img, save_path)
        assert mat.ndim == 2, 'mat should be 2D, where mat.size(0) is the number of data points'
        make_mat(mat, save_path)
        append_pbtxt(metadata, label_img,
                     self._get_file_writer().get_logdir(), subdir, global_step, tag)
    def add_pr_curve(self, tag, labels, predictions, global_step=None,
                     num_thresholds=127, weights=None, walltime=None):
        from Code_for_Experiment.Metrics.music_understanding_model.tensorboardX.tensorboardX.x2num import make_np
        labels, predictions = make_np(labels), make_np(predictions)
        self._get_file_writer().add_summary(
            pr_curve(tag, labels, predictions, num_thresholds, weights),
            global_step, walltime)
    def add_pr_curve_raw(self, tag, true_positive_counts,
                         false_positive_counts,
                         true_negative_counts,
                         false_negative_counts,
                         precision,
                         recall,
                         global_step=None,
                         num_thresholds=127,
                         weights=None,
                         walltime=None):
        self._get_file_writer().add_summary(
            pr_curve_raw(tag,
                         true_positive_counts,
                         false_positive_counts,
                         true_negative_counts,
                         false_negative_counts,
                         precision,
                         recall,
                         num_thresholds,
                         weights),
            global_step,
            walltime)
    def add_custom_scalars_multilinechart(self, tags, category='default', title='untitled'):
        layout = {category: {title: ['Multiline', tags]}}
        self._get_file_writer().add_summary(custom_scalars(layout))
    def add_custom_scalars_marginchart(self, tags, category='default', title='untitled'):
        assert len(tags) == 3
        layout = {category: {title: ['Margin', tags]}}
        self._get_file_writer().add_summary(custom_scalars(layout))
    def add_custom_scalars(self, layout):
        self._get_file_writer().add_summary(custom_scalars(layout))
    def add_mesh(self, tag, vertices, colors=None, faces=None, config_dict=None, global_step=None, walltime=None):
        self._get_file_writer().add_summary(mesh(tag, vertices, colors, faces, config_dict), global_step, walltime)
    def close(self):
        if self.all_writers is None:
            return
        for writer in self.all_writers.values():
            writer.flush()
            writer.close()
        self.file_writer = self.all_writers = None
    def flush(self):
        if self.all_writers is None:
            return
        for writer in self.all_writers.values():
            writer.flush()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()