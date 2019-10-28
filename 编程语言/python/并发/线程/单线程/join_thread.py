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
    logging.info("Main: before creating thread")
    x = threading.Thread(target=thread_function, args=(1,), daemon=False)
    logging.info("Main: before running thread")
    x.start()
    logging.info("Main: wait for the thread to finish")
    x.join()
    logging.info("Main: all done")

# 输出
19:59:01: Main: before creating thread
19:59:01: Main: before running thread
19:59:01: Thread 1: starting
19:59:01: Main: wait for the thread to finish
19:59:03: Thread 1: finishing
19:59:03: Main: all done
