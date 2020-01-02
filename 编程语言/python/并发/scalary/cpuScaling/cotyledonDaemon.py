root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# cat cotyledonDaemon.py 
import threading
import time
import cotyledon

class PrinterService(cotyledon.Service):
    name = "printer"
    def __init__(self, worker_id):
        super(PrinterService, self).__init__(worker_id)
        self._shutdown = threading.Event()

    def run(self):
        while not self._shutdown.is_set():
            print('Doing stuff')
            time.sleep(1)

    def terminate(self):
        self._shutdown.set()

manager = cotyledon.ServiceManager()
manager.add(PrinterService, 2)
manager.run()
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# 
