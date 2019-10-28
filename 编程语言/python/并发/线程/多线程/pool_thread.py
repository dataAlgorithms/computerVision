from concurrent.futures import ThreadPoolExecutor

import logging
import threading
import time

def thread_function(name):
    logging.info("Thread %s: starting", name)
    time.sleep(2)
    logging.info("Thread %s: finishing", name)

if __name__ == '__main__':
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
       datefmt="%H:%M:%S")

    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(thread_function, range(3))

# 输出
root@ubuntu:/home/zp/workspace/pyLearn/thread# python threadPool.py 
20:19:22: Thread 0: starting
20:19:22: Thread 1: starting
20:19:22: Thread 2: starting
20:19:24: Thread 0: finishing
20:19:24: Thread 1: finishing
20:19:24: Thread 2: finishing
