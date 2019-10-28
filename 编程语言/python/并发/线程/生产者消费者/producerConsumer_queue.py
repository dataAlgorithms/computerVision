import concurrent.futures
import logging
import queue
import random
import threading
import time

def producer(queue, event):
    """Pretend we're getting a number from the network."""
    while not event.is_set():
        message = random.randint(1, 5)
        logging.info("Producer got message: %s", message)
        queue.put(message)

    logging.info("Producer received event. Exiting")

def consumer(queue, event):
    """Pretend we're saving a number in the database."""
    while not event.is_set() or not queue.empty():
        message = queue.get()
        logging.info(
            "Consumer storing message: %s (size=%d)", message, queue.qsize()
        )

    logging.info("Consumer received event. Exiting")

class Pipeline(queue.Queue):
    def __init__(self):
        super().__init__(maxsize=2)

    def get_message(self, name):
        logging.debug("%s:about to get from queue", name)
        value = self.get()
        logging.debug("%s:got %d from queue", name, value)
        return value

    def set_message(self, value, name):
        logging.debug("%s:about to add %d to queue", name, value)
        self.put(value)
        logging.debug("%s:added %d to queue", name, value)

if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    pipeline = queue.Queue(maxsize=2)
    event = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(producer, pipeline, event)
        executor.submit(consumer, pipeline, event)

        time.sleep(0.1)
        logging.info("Main: about to set event")
        event.set()

# 输出
...
17:00:47: Producer got message: 3
17:00:47: Consumer storing message: 4 (size=1)
17:00:47: Producer got message: 1
17:00:47: Consumer storing message: 5 (size=1)
17:00:47: Producer got message: 1
17:00:47: Consumer storing message: 3 (size=1)
17:00:47: Producer got message: 1
17:00:47: Consumer storing message: 1 (size=1)
17:00:47: Producer got message: 5
17:00:47: Consumer storing message: 1 (size=1)
17:00:47: Producer got message: 1
17:00:47: Consumer storing message: 1 (size=1)
17:00:47: Producer got message: 3
17:00:47: Consumer storing message: 5 (size=1)
17:00:47: Producer got message: 4
17:00:47: Consumer storing message: 1 (size=1)
17:00:47: Producer got message: 2
17:00:47: Consumer storing message: 3 (size=1)
17:00:47: Producer got message: 4
17:00:47: Consumer storing message: 4 (size=1)
17:00:47: Producer got message: 1
17:00:47: Consumer storing message: 2 (size=1)
17:00:47: Producer got message: 4
17:00:47: Consumer storing message: 4 (size=1)
17:00:47: Producer got message: 1
17:00:47: Consumer storing message: 1 (size=1)
17:00:47: Producer got message: 3
17:00:47: Consumer storing message: 4 (size=1)
17:00:47: Producer got message: 4
17:00:47: Consumer storing message: 1 (size=1)
17:00:47: Producer got message: 2
17:00:47: Consumer storing message: 3 (size=1)
17:00:47: Producer got message: 5
17:00:47: Main: about to set event
17:00:47: Consumer storing message: 4 (size=1)
17:00:47: Producer got message: 4
17:00:47: Consumer storing message: 2 (size=1)
17:00:47: Producer received event. Exiting
17:00:47: Consumer storing message: 5 (size=1)
17:00:47: Consumer storing message: 4 (size=0)
17:00:47: Consumer received event. Exiting

问题:
不清楚如何控制输出条数??
