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
    x = threading.Thread(target=thread_function, args=(1,), daemon=True)
    logging.info("Main: before running thread")
    x.start()
    logging.info("Main: wait for the thread to finish")
    # x.join()
    logging.info("Main: all done")

# 输出
root@ubuntu:/home/zp/workspace/pyLearn/thread# python daemon_thread.py 
19:53:44: Main: before creating thread
19:53:44: Main: before running thread
19:53:44: Thread 1: starting
19:53:44: Main: wait for the thread to finish
19:53:44: Main: all done
