import threading
import random
import logging
from concurrent.futures import ThreadPoolExecutor

SENTINEL = object()

class Pipeline:
    """
    Class to allow a single element pipeline between producer and consumer.
    """
    def __init__(self):
        self.message = 0
        self.producer_lock = threading.Lock()
        self.consumer_lock = threading.Lock()
        self.consumer_lock.acquire()

    def get_message(self, name):
        self.consumer_lock.acquire()
        message = self.message
        self.producer_lock.release()
        return message

    def set_message(self, message, name):
        self.producer_lock.acquire()
        self.message = message
        self.consumer_lock.release()

def producer(pipeline):
    """Pretend we're getting a message from the network."""
    for index in range(10):
        message = random.randint(1, 101)
        logging.info("Producer got message: %s", message)
        pipeline.set_message(message, "Producer")

    # Send a sentinel message to tell consumer we're done
    pipeline.set_message(SENTINEL, "Producer")

def consumer(pipeline):
    """Pretend we're saving a number in the database."""
    message = 0
    while message is not SENTINEL:
        message = pipeline.get_message("Consumer")
        if message is not SENTINEL:
            logging.info("Consumer storing message: %s", message)

if __name__ == '__main__':
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
           datefmt='%H:%M:%S')
    logging.getLogger().setLevel(logging.DEBUG)
    pipeline = Pipeline()
    with ThreadPoolExecutor(max_workers=2) as executor:
       executor.submit(producer, pipeline)
       executor.submit(consumer, pipeline)

# 输出
16:45:49: Producer got message: 18
16:45:49: Producer got message: 13
16:45:49: Consumer storing message: 18
16:45:49: Producer got message: 62
16:45:49: Consumer storing message: 13
16:45:49: Producer got message: 20
16:45:49: Consumer storing message: 62
16:45:49: Producer got message: 23
16:45:49: Consumer storing message: 20
16:45:49: Producer got message: 81
16:45:49: Consumer storing message: 23
16:45:49: Producer got message: 56
16:45:49: Consumer storing message: 81
16:45:49: Producer got message: 3
16:45:49: Consumer storing message: 56
16:45:49: Producer got message: 17
16:45:49: Consumer storing message: 3
16:45:49: Producer got message: 67
16:45:49: Consumer storing message: 17
16:45:49: Consumer storing message: 67
