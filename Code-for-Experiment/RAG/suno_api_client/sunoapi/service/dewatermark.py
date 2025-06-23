import multiprocessing
import queue
import time
import os
from sunoapi.encodec import EncoedecProcessor
import sunoapi.config_loader as config
import logging
from sunoapi.service.remote_file import download_file_from_url, upload_file
from Code_for_Experiment.RAG.suno_api_client.sunoapi.utils.string_utils import generate_random_string

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaskProcessor(multiprocessing.Process):
    def __init__(self, task_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue, error_queue: multiprocessing.Queue, device_id: int):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.error_queue = error_queue
        self.stop_event = multiprocessing.Event()
        self.device = f"cuda:{device_id}"

    def run(self):
        if config.config["dewatermark"]["use_encodec_dewatermark"]:
            processor = EncoedecProcessor(device=self.device)
        else:
            processor = None
        logger.info("processor ready:%s", self.device)
        while not self.stop_event.is_set():
            try:
                task = self.task_queue.get(block=True, timeout=1)
                if task is None:
                    break
                result = self.process_task(processor, task)
                if self.result_queue:
                    self.result_queue.put(result)
            except queue.Empty:
                pass
            except Exception as e:
                logger.error("process task error:%s", e)
                self.error_queue.put(f"api_call:{e},{task}")

    def process_task(self, processor: EncoedecProcessor, task: dict):
        if config.config["dewatermark"]["use_encodec_dewatermark"]:
            processed_pathes = []
            logger.info("process list:%s", task)
            for path in task["path"]:
                processed_path = path + "_dewatermark.mp3"
                processor.process_file(
                    audio_file=path,
                    out_file=processed_path
                )
                if os.path.exists(processed_path):
                    processed_pathes.append(processed_path)
            res = self.submit_result(processed_pathes, task["args"])
            for processed_path in processed_pathes:
                if os.path.exists(processed_path):
                    os.remove(processed_path)
        else:
            res = self.submit_result(task["path"], task["args"])
        for path in task["path"]:
            if os.path.exists(path):
                os.remove(path)
        return res

    def submit_result(self, pathes: list[str], args: dict):
        rand_id = generate_random_string(16)
        mid = args["mid"]
        logger.info("submit result:%s mid:%s", pathes, mid)
        res = []
        for i, path in enumerate(pathes):
            mp3_oss_address = f"genaudio/{rand_id}/{mid}_{i}.mp3"
            upload_file(path, mp3_oss_address)
            res.append(mp3_oss_address)
        res = {"org_music_oss": config.config.oss.host + "/" + res[0], "mid": mid, "server": ""}
        logger.info("submit result success:%s", res)
        return res

    def stop(self):
        self.stop_event.set()
        self.task_queue.put(None)

class TaskManager:
    def __init__(self, task_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue, error_queue: multiprocessing.Queue, device_list: list[int]):
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.error_queue = error_queue
        self.device_list = device_list
        self.processors = [TaskProcessor(self.task_queue, self.result_queue, self.error_queue, device) for device in device_list]

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