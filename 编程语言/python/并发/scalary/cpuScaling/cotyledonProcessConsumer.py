root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# python37 cotyledonProcessConsumer.py 
I am Worker: 0 PID: 40849 and I print 0
I am Worker: 0 PID: 40849 and I print 1
I am Worker: 1 PID: 40851 and I print 2
I am Worker: 0 PID: 40849 and I print 3
I am Worker: 1 PID: 40851 and I print 4
I am Worker: 0 PID: 40849 and I print 5
I am Worker: 1 PID: 40851 and I print 6
I am Worker: 0 PID: 40849 and I print 7
I am Worker: 1 PID: 40851 and I print 8
I am Worker: 0 PID: 40849 and I print 9
I am Worker: 1 PID: 40851 and I print 10
I am Worker: 0 PID: 40849 and I print 11
^Croot@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# cat cotyledonProcessConsumer.py 
import multiprocessing
import time
import cotyledon

class Manager(cotyledon.ServiceManager):
    def __init__(self):
        super(Manager, self).__init__()
        queue = multiprocessing.Manager().Queue()
        self.add(ProducerService, args=(queue,))
        self.add(PrinterService, args=(queue,), workers=2)

class ProducerService(cotyledon.Service):
    def __init__(self, worker_id, queue):
        super(ProducerService, self).__init__(worker_id)
        self.queue = queue

    def run(self):
        i = 0
        while True:
            self.queue.put(i)
            i += 1
            time.sleep(1)

class PrinterService(cotyledon.Service):
    name = "printer"
    def __init__(self, worker_id, queue):
        super(PrinterService, self).__init__(worker_id)
        self.queue = queue

    def run(self):
        while True:
            job = self.queue.get(block=True)
            print("I am Worker: %d PID: %d and I print %s" % (self.worker_id, self.pid, job))

Manager().run()
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# 
