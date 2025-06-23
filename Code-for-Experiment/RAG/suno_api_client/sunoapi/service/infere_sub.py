import multiprocessing
import queue
import os
import time
import json
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import pika
from sunoapi.service.mqc_helper import SUBQ_NAME, get_mq_cnx
from sunoapi.service.result_pub import send_push
from sunoapi.service.remote_file import download_file_from_url, upload_file
from Code_for_Experiment.RAG.suno_api_client.sunoapi.utils.string_utils import generate_random_string
import sunoapi.service.dewatermark
import sunoapi.service.suno_api_call
import sunoapi.config_loader

config = sunoapi.config_loader.config
current_dir = os.path.dirname(os.path.abspath(__file__))
task_queue = multiprocessing.Queue()
result_queue = multiprocessing.Queue()
upload_queue = multiprocessing.Queue()
error_queue = multiprocessing.Queue()
suno = sunoapi.service.suno_api_call.TaskManager(task_queue, result_queue, error_queue, config.suno.num_work)
dewm = sunoapi.service.dewatermark.TaskManager(result_queue, upload_queue, error_queue, device_list=config.dewatermark.device_list)
suno.start()
dewm.start()

def read_all_from_queue(q):
    results = []
    while True:
        try:
            item = q.get_nowait()
            results.append(item)
        except queue.Empty:
            break
    return results

def upload_tasks(queue:multiprocessing.Queue):
    msg = read_all_from_queue(queue)
    if len(msg) == 0:
        return
    logger.info(" Done! Sending succ type msg...")
    data = {"type":"succ","msg":msg}
    send_push(data)
    logger.info("upload:%s", data)

def upload_error(queue:multiprocessing.Queue):
    msg = read_all_from_queue(queue)
    if len(msg) == 0:
        return
    logger.info(" Done! Sending error type msg...")
    data = {"type":"fail","error":msg}
    send_push(data)
    logger.info("error:%s", data)

def upload_process_func(upload_queue, error_queue):
    logger.info("upload_process_func:start")
    while True:
        try:
            upload_tasks(upload_queue)
        except Exception as e:
            logger.error("upload_process_func error: %s",e)
        try:
            upload_error(error_queue)
        except Exception as e:
            logger.error("upload_process_func error: %s",e)
        time.sleep(4)

def gen_audio(tasks:list[dict]):
    for task in tasks:
        logger.info("task:%s", task)
        suno.add_task(task)

upload_process = multiprocessing.Process(target=upload_process_func, args=(upload_queue,error_queue))
upload_process.start()

def callback(
    channel: pika.channel.Channel,
    method: pika.spec.Basic.Deliver,
    properties: pika.spec.BasicProperties,
    body: bytes):
    pid = os.getpid()
    data = json.loads(body)
    logger.info(" [x]-%d ATTN: Sending the ok msg immediately!",pid)
    result = []
    for it in data:
        result.append({"mid":it["mid"],"oss":"","server":"end"})
    send_push({"type":"ok","msg":result})
    logger.info(" [x]-%d Start merge data...", pid)
    gen_audio(data)

def subscribe():
    cnx_rec_max_t = 4
    cnx_rec_times = 0
    while cnx_rec_times <= cnx_rec_max_t:
        try:
            connection = get_mq_cnx()
            chan_ip = connection.channel()
            chan_ip.queue_declare(queue=SUBQ_NAME, durable=True)
            chan_ip.basic_qos(prefetch_count=1)
            chan_ip.basic_consume(
                queue=SUBQ_NAME,
                auto_ack=True,
                on_message_callback=callback)
            chan_ip.start_consuming()
        except pika.exceptions.ConnectionClosedByBroker:
            break
        except pika.exceptions.AMQPChannelError:
            break
        except pika.exceptions.AMQPConnectionError:
            cnx_rec_times += 1
            time.sleep(2)
            continue

def start():
    logger.info("start")
    workers_num = 8
    mpp = multiprocessing.Pool(processes=workers_num)
    for i in range(workers_num):
        mpp.apply_async(subscribe)
    mpp.close()
    mpp.join()