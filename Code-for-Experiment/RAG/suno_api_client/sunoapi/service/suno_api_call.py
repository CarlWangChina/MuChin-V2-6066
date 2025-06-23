import multiprocessing
import queue
import time
from sunoapi.api import generate_music, download_result
import sunoapi.config_loader as config
import logging
import traceback

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaskProcessor(multiprocessing.Process):
    def __init__(self, task_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue = None, error_queue: multiprocessing.Queue = None):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.error_queue = error_queue
        self.stop_event = multiprocessing.Event()

    def run(self):
        while not self.stop_event.is_set():
            try:
                task = self.task_queue.get(block=True, timeout=1)
                if task is None:
                    break
                result = self.process_task(task)
                if self.result_queue:
                    self.result_queue.put(result)
            except queue.Empty:
                pass
            except Exception as e:
                logger.error("process task error:%s", e)
                if self.error_queue:
                    self.error_queue.put({"mid": task["mid"], "err_msg": f"api call:{task} {traceback.format_exc()}"})

    def process_task(self, task):
        music_result = generate_music(task["title"], task["style"], task["lyric"], mv=config.config["suno"]["model"])
        out_dir = "/tmp/"
        logger.info("Downloading:%s", music_result)
        return {"path": download_result(music_result, out_dir), "args": task}

    def stop(self):
        self.stop_event.set()
        self.task_queue.put(None)

class TaskManager:
    def __init__(self, task_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue = None, error_queue: multiprocessing.Queue = None, num_processes: int = 1):
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.error_queue = error_queue
        self.num_processes = num_processes
        self.processors = [TaskProcessor(self.task_queue, self.result_queue, self.error_queue) for _ in range(num_processes)]

    def start(self):
        for p in self.processors:
            p.start()

    def stop(self):
        for p in self.processors:
            p.stop()
        for p in self.processors:
            p.join()

    def add_task(self, task):
        self.task_queue.put(task)