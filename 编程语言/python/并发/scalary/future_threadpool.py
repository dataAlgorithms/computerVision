root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# cat future_threadpool.py 
from concurrent import futures
import random

def compute():
    return sum(
    [random.randint(1, 100) for i in range(1000000)])

with futures.ThreadPoolExecutor(max_workers=8) as executor:
    futs = [executor.submit(compute) for _ in range(8)]
    results = [f.result() for f in futs]
    print("Results: %s" % results)
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# python37 future_threadpool.py 
Results: [50477202, 50527542, 50545536, 50508597, 50501607, 50508008, 50532423, 50500573]
