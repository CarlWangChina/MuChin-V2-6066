import multiprocessing
import random
import time

def producer(queue, max_items):
    for _ in range(max_items):
        item = random.randint(1, 100)
        print("Producing item:", item)
        queue.put(item)
        time.sleep(random.random())
    queue.put(None)

def consumer(queue):
    while True:
        item = queue.get()
        if item is None:
            break
        print("Consuming item:", item)
        time.sleep(random.random())

if __name__ == "__main__":
    queue = multiprocessing.Queue()
    max_items = 10
    producer_process = multiprocessing.Process(target=producer, args=(queue, max_items))
    consumer_process = multiprocessing.Process(target=consumer, args=(queue,))
    producer_process.start()
    consumer_process.start()
    producer_process.join()
    consumer_process.join()