import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sunoapi.service.dewatermark import *
import multiprocessing
import time

if __name__ == "__main__":
    task_queue = multiprocessing.Queue()
    res_queue = multiprocessing.Queue()
    err_queue = multiprocessing.Queue()
    manager = TaskManager(task_queue, res_queue, err_queue, device_list=[0, 1, 2, 3])
    manager.start()
    manager.add_task({
        "path": ['/home/dev-contributor/sunoapi/tests/outputs/test_gen_queue//9b55f6af-5cf1-4520-b999-66cf72b3a7e4.mp3', '/home/dev-contributor/sunoapi/tests/outputs/test_gen_queue//9793b6ae-bdfe-48ff-81cf-7c1302f74565.mp3'],
        "args": {"mid": "10000000"}
    })
    print("push task success")
    time.sleep(2)
    while True:
        try:
            print(res_queue.get(True, 1))
        except:
            break
    print("success")