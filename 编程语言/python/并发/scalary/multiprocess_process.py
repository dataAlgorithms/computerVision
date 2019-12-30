root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# cat multiprocess_process.py 
import random
import multiprocessing

def compute(results):
    results.append(sum(
    [random.randint(1, 100) for i in range(1000000)]))

with multiprocessing.Manager() as manager:
    results = manager.list()
    workers = [multiprocessing.Process(target=compute, args=(results,)) for x in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    print("Results: %s" % results)
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# python37 multiprocess_process.py 
Results: [50499228, 50504964, 50484402, 50436302, 50529100, 50443367, 50474507, 50510043]
