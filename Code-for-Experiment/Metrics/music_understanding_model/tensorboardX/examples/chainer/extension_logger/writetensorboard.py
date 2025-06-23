import json
import os
import shutil
import tempfile
import six
from chainer import reporter
from chainer import serializer as serializer_module
from chainer.training import extension
from chainer.training import trigger as trigger_module

class LogTensorboard(extension.Extension):
    def __init__(self, keys=None, trigger=(1, 'epoch'), postprocess=None, log_name='log', logger=None):
        self._keys = keys
        self._trigger = trigger_module.get_trigger(trigger)
        self._postprocess = postprocess
        self._log_name = log_name
        self._log = []
        self._logger = logger
        self._init_summary()

    def __call__(self, trainer):
        keys = self._keys
        observation = trainer.observation
        summary = self._summary
        if keys is None:
            summary.add(observation)
        else:
            summary.add({k: observation[k] for k in keys if k in observation})
        for k, v in observation.items():
            self._logger.add_scalar(k, observation[k], trainer.updater.iteration)
        if self._trigger(trainer):
            stats = self._summary.compute_mean()
            stats_cpu = {}
            for name, value in six.iteritems(stats):
                stats_cpu[name] = float(value)
            updater = trainer.updater
            stats_cpu['epoch'] = updater.epoch
            stats_cpu['iteration'] = updater.iteration
            stats_cpu['elapsed_time'] = trainer.elapsed_time
            if self._postprocess is not None:
                self._postprocess(stats_cpu)
            self._log.append(stats_cpu)
            if self._log_name is not None:
                log_name = self._log_name.format(**stats_cpu)
                fd, path = tempfile.mkstemp(prefix=log_name, dir=trainer.out)
                with os.fdopen(fd, 'w') as f:
                    json.dump(self._log, f, indent=4)
                new_path = os.path.join(trainer.out, log_name)
                shutil.move(path, new_path)
            self._init_summary()

    @property
    def log(self):
        return self._log

    def serialize(self, serializer):
        if hasattr(self._trigger, 'serialize'):
            self._trigger.serialize(serializer['_trigger'])
        if isinstance(serializer, serializer_module.Serializer):
            log = json.dumps(self._log)
            serializer('_log', log)
        else:
            log = serializer('_log', '')
            self._log = json.loads(log)

    def _init_summary(self):
        self._summary = reporter.DictSummary()