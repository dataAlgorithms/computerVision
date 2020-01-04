root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# python37 future_processpool.py 
Results: [509201, 504775, 504086, 503201, 504812, 496241, 507521, 503975]
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# cat future_processpool.py 
from concurrent import futures
import random

def compute():
    return sum([random.randint(1, 100) for i in range(10000)])

with futures.ProcessPoolExecutor() as executor:
    futs = [executor.submit(compute) for _ in range(8)]

results = [f.result() for f in futs]
print('Results: %s' % results)
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# 
