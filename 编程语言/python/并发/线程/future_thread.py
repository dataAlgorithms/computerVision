root@ubuntu:/home/zhouping/anaconda3/nn/scalingPy# python37 future_thread.py 
Results: [50495536, 50555665, 50502269, 50497127, 50521471, 50562390, 50490452, 50499269]
root@ubuntu:/home/zhouping/anaconda3/nn/scalingPy# cat future_thread.py 
from concurrent import futures
import random

def compute():
    return sum([random.randint(1, 100) for i in range(1000000)])

with futures.ThreadPoolExecutor(max_workers=8) as executor:
    futs = [executor.submit(compute) for _ in range(8)]

results = [f.result() for f in futs]
print('Results: %s' % results)
root@ubuntu:/home/zhouping/anaconda3/nn/scalingPy# 
